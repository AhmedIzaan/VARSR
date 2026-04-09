import os
import sys
import glob
import argparse
import numpy as np
import yaml
from PIL import Image
import torch.nn.functional as F
import safetensors.torch
import time
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
import dist
import torch
from torchvision import transforms
import torch.utils.checkpoint
from utils import arg_util, misc
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import CLIPTextModel, CLIPVisionModel, CLIPTokenizer, CLIPImageProcessor
from myutils.wavelet_color_fix import wavelet_color_fix, adain_color_fix
from dataloader.testdataset import TestDataset
import math
from torch.utils.data import DataLoader
from torchvision import transforms
import pyiqa
from skimage import io
from models import VAR_RoPE, VQVAE, build_var


def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB images to YCbCr images (PyTorch version).
    It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    Args:
        img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
         y_only (bool): Whether to only return Y channel. Default: False.
    Returns:
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img
def img2tensor(img):
    img = (img / 255.).astype('float32')
    if img.ndim ==2:
        img = np.expand_dims(np.expand_dims(img, axis = 0),axis=0)
    else:
        img = np.transpose(img, (2, 0, 1))  # C, H, W
        img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img, dtype=np.float32)
    tensor = torch.from_numpy(img)
    return tensor
def numpy_to_pil(images: np.ndarray):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images
def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy image.
    """
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images

logger = get_logger(__name__, log_level="INFO")

def main(args: arg_util.Args, output_tag: str = 'VARPrediction'):
    vae_ckpt =  args.vae_model_path
    var_ckpt = args.var_test_path
    args.depth = 24

    # Build on CPU, load weights, then move to GPU — avoids NVRTC kernel
    # compilation during init_weights which can fail on some CUDA environments.
    vae, var = build_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4, controlnet_depth=args.depth,        # hard-coded VQVAE hyperparameters
        device='cpu', patch_nums=args.patch_nums, control_patch_nums=args.patch_nums,
        num_classes=1 + 1, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
    )
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu')['trainer']['vae_local'], strict=True)
    model_state = torch.load(var_ckpt, map_location='cpu')
    var.load_state_dict(model_state['trainer']['var_wo_ddp'], strict=True)
    device = dist.get_device()
    vae.to(device).eval()
    var.to(device).eval()

    folders = os.listdir("testset/")
    val_set = []
    for folder in folders:
        dataset_val = TestDataset("testset/" + folder, image_size=args.data_load_reso, tokenizer=None, resize_bak=True)
        ld_val = DataLoader(
            dataset_val, num_workers=0, pin_memory=True,
            batch_size=round(args.batch_size), sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
            shuffle=False, drop_last=False,
        )
        val_set.append(ld_val)

    for ld_val in val_set:
        for batch in ld_val:
            lr_inp = batch["conditioning_pixel_values"].to(args.device, non_blocking=True)
            label_B = batch["label_B"].to(args.device, non_blocking=True)
            B = lr_inp.shape[0]

            with torch.inference_mode():
                with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                    recon_B3HW = var.autoregressive_infer_cfg(
                        B=B, cfg=args.cfg, top_k=args.top_k, top_p=args.top_p,
                        diff_temp=args.diff_temp, g_seed=args.infer_seed,
                        text_hidden=None, lr_inp=lr_inp, negative_text=None,
                        label_B=label_B, lr_inp_scale=None, more_smooth=False,
                    )
                    recon_B3HW = numpy_to_pil(pt_to_numpy(recon_B3HW))

            for idx in range(B):
                image = recon_B3HW[idx]
                if args.color_fix != 'none':
                    validation_image = Image.open(batch['path'][idx].replace("/HR/", "/LR/").replace("_HR.png", "_LR4.png")).convert("RGB")
                    validation_image = validation_image.resize((512, 512))
                    if args.color_fix == 'wavelet':
                        image = wavelet_color_fix(image, validation_image)
                    else:
                        image = adain_color_fix(image, validation_image)

                folder_path, ext_path = os.path.split(batch['path'][idx])
                output_name = folder_path.replace("/LR", f"/{output_tag}/").replace("/HR", f"/{output_tag}/")
                os.makedirs(output_name, exist_ok=True)
                image.save(os.path.join(output_name, ext_path))
    return True


def metrics(output_tag: str = 'VARPrediction'):
    dir = "testset/"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    folders = os.listdir("testset/")
    img_preproc = transforms.Compose([transforms.ToTensor()])

    psnr_metric = pyiqa.create_metric('psnr', device=device)
    ssim_metric = pyiqa.create_metric('ssim', device=device)
    fid_metric = pyiqa.create_metric('fid', device=device)
    maniqa_metric = pyiqa.create_metric('maniqa', device=device)
    lpips_iqa_metric = pyiqa.create_metric('lpips', device=device)
    clipiqa_iqa_metric = pyiqa.create_metric('clipiqa', device=device)
    musiq_iqa_metric = pyiqa.create_metric('musiq', device=device)
    dists_iqa_metric = pyiqa.create_metric('dists', device=device)
    niqe_iqa_metric = pyiqa.create_metric('niqe', device=device)

    all_results = {}
    for folder in folders:
        print(f"\n--- {folder} ---")
        gt_img_paths = []
        gt_img_paths.extend(sorted(glob.glob(f'{dir}/{folder}/HR/*.JPEG')))
        gt_img_paths.extend(sorted(glob.glob(f'{dir}/{folder}/HR/*.png')))

        real_image_folder = f"{dir}/{folder}/HR"
        generated_image_folder = real_image_folder.replace("/HR", f"/{output_tag}")

        psnr_folder, ssim_folder, lpips_iqa, musiq_iqa, maniqa_iqa, clip_iqa, dists_score, niqe_score = [], [], [], [], [], [], [], []

        for gt_img_path in gt_img_paths:
            prediction_img_path = gt_img_path.replace("/HR/", f"/{output_tag}/")
            if not os.path.exists(prediction_img_path):
                print(f"  [skip] missing: {prediction_img_path}")
                continue

            # Get prediction size and resize GT to match — model outputs 512×512
            # but original HR images may be at full camera resolution (e.g. 1000×1400)
            pred_img = Image.open(prediction_img_path).convert('RGB')
            pred_w, pred_h = pred_img.size

            gt_raw = io.imread(gt_img_path)
            if gt_raw.shape[1] != pred_w or gt_raw.shape[0] != pred_h:
                gt_raw = np.array(Image.fromarray(gt_raw).resize((pred_w, pred_h), Image.BICUBIC))

            # Save resized GT to a temp path for metrics that take file paths
            gt_resized_path = gt_img_path.replace("/HR/", f"/{output_tag}_gtresized/")
            os.makedirs(os.path.dirname(gt_resized_path), exist_ok=True)
            Image.fromarray(gt_raw).save(gt_resized_path)

            img1 = rgb2ycbcr_pt(img2tensor(gt_raw), y_only=True).to(torch.float64)
            img2 = rgb2ycbcr_pt(img2tensor(io.imread(prediction_img_path)), y_only=True).to(torch.float64)
            img1, img2 = torch.squeeze(img1), torch.squeeze(img2)

            psnr_folder.append(psnr_metric(img1.unsqueeze(0).unsqueeze(0), img2.unsqueeze(0).unsqueeze(0)))
            ssim_folder.append(ssim_metric(img1.unsqueeze(0).unsqueeze(0), img2.unsqueeze(0).unsqueeze(0)))
            lpips_iqa.append(lpips_iqa_metric(prediction_img_path, gt_resized_path))
            clip_iqa.append(clipiqa_iqa_metric(prediction_img_path))
            musiq_iqa.append(musiq_iqa_metric(prediction_img_path))
            maniqa_iqa.append(maniqa_metric(prediction_img_path))
            dists_score.append(dists_iqa_metric(prediction_img_path, gt_resized_path))
            niqe_score.append(niqe_iqa_metric(prediction_img_path))

        if not psnr_folder:
            print("  No predictions found, skipping.")
            continue

        m_psnr   = (sum(psnr_folder) / len(psnr_folder)).item()
        m_ssim   = (sum(ssim_folder) / len(ssim_folder)).item()
        m_lpips  = (sum(lpips_iqa) / len(lpips_iqa)).item()
        m_dists  = (sum(dists_score) / len(dists_score)).item()
        m_niqe   = (sum(niqe_score) / len(niqe_score)).item()
        m_clip   = (sum(clip_iqa) / len(clip_iqa)).item()
        m_musiq  = (sum(musiq_iqa) / len(musiq_iqa)).item()
        m_maniqa = (sum(maniqa_iqa) / len(maniqa_iqa)).item()
        fid_value = fid_metric(real_image_folder, generated_image_folder)

        print(f"  PSNR     = {m_psnr:.4f}")
        print(f"  SSIM     = {m_ssim:.4f}")
        print(f"  LPIPS    = {m_lpips:.4f}  (lower=better)")
        print(f"  DISTS    = {m_dists:.4f}  (lower=better)")
        print(f"  NIQE     = {m_niqe:.4f}  (lower=better)")
        print(f"  CLIP-IQA = {m_clip:.4f}")
        print(f"  MUSIQ    = {m_musiq:.4f}")
        print(f"  MANIQA   = {m_maniqa:.4f}")
        print(f"  FID      = {fid_value:.4f}  (lower=better)")

        all_results[folder] = dict(
            psnr=m_psnr, ssim=m_ssim, lpips=m_lpips, dists=m_dists,
            niqe=m_niqe, clip_iqa=m_clip, musiq=m_musiq, maniqa=m_maniqa, fid=fid_value,
        )

    return all_results



if __name__ == "__main__":
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    tag = f"VARPrediction_cfg{args.cfg}_tk{args.top_k}_tp{args.top_p}_dt{args.diff_temp}_{args.color_fix}"
    print(f"\n[VARSR] output_tag = {tag}")
    print(f"[VARSR] cfg={args.cfg}  top_k={args.top_k}  top_p={args.top_p}  diff_temp={args.diff_temp}  color_fix={args.color_fix}  seed={args.infer_seed}\n")
    main(args, output_tag=tag)
    results = metrics(output_tag=tag)
