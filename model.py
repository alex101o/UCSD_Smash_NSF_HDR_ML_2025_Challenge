import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
import os
from PIL import Image
import random
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import ToPILImage

from __future__ import annotations

import cv2
from scipy.ndimage import binary_fill_holes

import timm
from tqdm import tqdm
from argparse import ArgumentParser

from ultralytics import SAM

import io
import contextlib



## UNCOMMENT AND RUN THIS IF YOU DO NOT HAVE ULTRALYTICS OR CV2 PIP INSTALLED!!!
#pip install ultralytics, opencv-python

root = Path("hackdata/sentinel-beetles/")  # change to testing root path
val_df = pd.read_csv(root / "public_release" / "val.csv")

## general arguments for the model
def get_training_args(argv=None):
    p = ArgumentParser()

    # paths
    p.add_argument("--train_csv", type=str, default="hackdata/sentinel-beetles/public_release/train.csv")
    p.add_argument("--val_csv", type=str, default="hackdata/sentinel-beetles/public_release/val.csv")
    p.add_argument("--train_img_dir", type=str, default="training_images")
    p.add_argument("--val_img_dir", type=str, default="validation_images")
    p.add_argument("--save_dir", type=str, default="ckpts")

    # columns
    p.add_argument("--event_col", type=str, default="eventID")
    p.add_argument("--img_col", type=str, default="relative_img_loc")

    # targets
    p.add_argument("--spei30_col", type=str, default="SPEI_30d")
    p.add_argument("--spei1y_col", type=str, default="SPEI_1y")
    p.add_argument("--spei2y_col", type=str, default="SPEI_2y")

    # model / data
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--k_max", type=int, default=8)          # max images per event used
    p.add_argument("--batch_size", type=int, default=1)     # events per batch (keep small!)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)

    # optimization
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-5)        # good default for finetuning convnext_small
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_accum", type=int, default=1)     # increase if OOM
    p.add_argument("--freeze_backbone_epochs", type=int, default=1)  # stabilize early training

    return p.parse_args(argv)

## calculating the r^2 score
def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot

def evaluate_spei_r2(gts: np.ndarray, preds: np.ndarray) -> Tuple[float, float, float]:
    return (
        r2_score_np(gts[:, 0], preds[:, 0]),
        r2_score_np(gts[:, 1], preds[:, 1]),
        r2_score_np(gts[:, 2], preds[:, 2]),
    )

## grabs the .csv, where to look for the images, etc

class EventDataset(Dataset):
    """
    Each __getitem__ returns:
      x: (K, 3, H, W) tensor of beetle images for this event (padded if <K)
      mask: (K,) float mask where 1 = real image, 0 = padding
      y: (3,) float targets for the event
    """

    def __init__(self, csv_path: str, img_dir: str, event_col: str, img_col: str, target_cols: Tuple[str, str, str], tfm, k_max: int, train_mode: bool, seed: int = 0,):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.img_root = Path(img_dir)
        self.event_col = event_col
        self.img_col = img_col
        self.target_cols = target_cols
        self.tfm = tfm
        self.k_max = k_max
        self.train_mode = train_mode
        self.rng = random.Random(seed)

        # sanity columns
        for col in [event_col, img_col, *target_cols]:
            if col not in self.df.columns:
                raise KeyError(f"Missing column '{col}' in {csv_path}. Columns: {list(self.df.columns)}")

        # drop missing
        self.df = self.df.dropna(subset=[event_col, img_col, *target_cols]).reset_index(drop=True)

        
        # Build event -> row indices
        self.event_to_rows: Dict[str, List[int]] = {}
        for i in range(len(self.df)):
            ev = str(self.df.loc[i, self.event_col])
            self.event_to_rows.setdefault(ev, []).append(i)
        self.events = list(self.event_to_rows.keys())

    def __len__(self) -> int:
        return len(self.events)

    def _open_image(self, rel: str) -> Image.Image:
        rel = str(rel).lstrip("/").replace("\\", "/")
        p = self.img_root / rel
        if not p.exists():
            # fallback: basename only
            p2 = self.img_root / Path(rel).name
            if p2.exists():
                p = p2
            else:
                raise FileNotFoundError(f"Image not found: {p} (also tried {p2})")
        return Image.open(p).convert("RGB")
    
    def __getitem__(self, idx: int):
        ev = self.events[idx]
        rows = self.event_to_rows[ev]
    
        # targets from first row
        row0 = self.df.loc[rows[0]]
        y = torch.tensor([row0[c] for c in self.target_cols], dtype=torch.float32)
    
        # sample up to k_max
        if self.train_mode and len(rows) > self.k_max:
            chosen = self.rng.sample(rows, self.k_max)
        else:
            chosen = rows[:self.k_max]
    
        xs = []
    
        for r in chosen:
            rel_path = self.df.loc[r, self.img_col]
            img = self._open_image(rel_path)  # returns PIL or np array
    
            # MaskTransform expects np.ndarray (H x W x C)
            if isinstance(img, Image.Image):
                img = np.array(img)[:, :, ::-1]  # PIL RGB -> BGR for your MaskTransform
    
            x = self.tfm(img)  # returns Tensor [C, H, W]
            xs.append(x)
    
        n = len(xs)
        H = W = self.tfm.final_size if hasattr(self.tfm, "final_size") else xs[0].shape[1]
    
        # pad to k_max
        if n < self.k_max:
            pad = torch.zeros((self.k_max - n, 3, H, W), dtype=torch.float32)
            x = torch.cat([torch.stack(xs, dim=0), pad], dim=0)
        else:
            x = torch.stack(xs[:self.k_max], dim=0)
    
        # simple mask: 1 for real images, 0 for padded
        mask = torch.zeros((self.k_max,), dtype=torch.float32)
        mask[:n] = 1.0

        return x, mask, y

## defines the ConvNeXt Regressor to output 3 SPEI predictions (30d, 1y, 2y)
class EventConvNeXtRegressor(nn.Module):
    def __init__(self, backbone_name: str = "convnext_small"):
        super().__init__()
        self.backbone = timm.create_model("convnext_small", pretrained=True, num_classes=0, cache_dir=str(Path("timm_cache").resolve()),
)
        d = self.backbone.num_features
        self.head = nn.Linear(d, 6)  # 3 mu + 3 log_sigma

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x: (B, K, 3, H, W)
        mask: (B, K) 1 for real, 0 for padded
        """
        B, K, C, H, W = x.shape
        feats = self.backbone(x.view(B * K, C, H, W))  # (B*K, d)
        d = feats.shape[-1]
        feats = feats.view(B, K, d)             # (B, K, d)

        # masked mean pool
        m = mask.unsqueeze(-1)                  # (B, K, 1)
        denom = m.sum(dim=1).clamp_min(1.0)     # (B, 1)
        event_feat = (feats * m).sum(dim=1) / denom  # (B, d)

        out = self.head(event_feat)             # (B, 6)
        mu = out[:, :3]                         # (B, 3)
        sigma = F.softplus(out[:, 3:]) + 1e-6

        print("EventConvNeXtRegressor")
        return mu, sigma

## defining the Gaussian negative log-likelihood

def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    y, mu, sigma: (B, 3)
    Negative log-likelihood for independent Gaussian dims.
    """
    # 0.5*((y-mu)/sigma)^2 + log(sigma)
    return (0.5 * ((y - mu) / sigma).pow(2) + torch.log(sigma)).mean()

## masking function, uses SAM to segment beetles, normalize size turned into a PyTorch transformation

class MaskTransform:
    
    def __init__(self, model_path: str = "sam2.1_l.pt", final_size: int = 512, target_fraction: float = 0.67, device="cuda"):
        self.model = SAM(model_path)
        self.final_size = final_size
        self.target_fraction = target_fraction

    def __call__(self, img: np.ndarray) -> torch.Tensor:        #img: H x W x C numpy array (BGR), returns: C x H x W torch tensor
        
        H, W = img.shape[:2]
        attempt_counter = 0
        best_mask = None

        # try a few times to get a valid mask
        while attempt_counter < 10 and best_mask is None:
            points = [[W * (0.5+random.uniform(-0.1, 0.1)), H * (0.35+random.uniform(-0.1, 0.1))], [5, 5]]
            labels = [1, 0]

            with contextlib.redirect_stdout(io.StringIO()):
                results = self.model(img, points=points, labels=labels)
            r = results[0]

            if r.masks is None:
                attempt_counter += 1
                continue

            masks = r.masks.data.cpu().numpy()
            best_area = 0
            for m in masks:
                area = m.sum()
                area_ratio = area / (W * H)
                if 0.25 < area_ratio < 0.75:
                    if area > best_area:
                        best_area = area
                        best_mask = m

            attempt_counter += 1

        if best_mask is None:
            # fallback: return resized original
            img_resized = cv2.resize(img, (self.final_size, self.final_size))
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            return transforms.ToTensor()(img_resized)

        # smoothing + polishing
        mask = binary_fill_holes(best_mask).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            img_resized = cv2.resize(img, (self.final_size, self.final_size))
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            return transforms.ToTensor()(img_resized)

        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()

        beetle = img[ymin:ymax+1, xmin:xmax+1]
        beetle_mask = mask[ymin:ymax+1, xmin:xmax+1]

        # scale normalization
        h, w = beetle.shape[:2]
        target_size = int(self.target_fraction * self.final_size)
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        beetle = cv2.resize(beetle, (new_w, new_h), interpolation=cv2.INTER_AREA)
        beetle_mask = cv2.resize(beetle_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # white square padding
        side = max(new_h, new_w)
        square = np.ones((side, side, 3), dtype=np.uint8) * 255
        yoff = (side - new_h) // 2
        xoff = (side - new_w) // 2
        square[yoff:yoff+new_h, xoff:xoff+new_w][beetle_mask > 0] = beetle[beetle_mask > 0]

        # final resize
        square = cv2.resize(square, (self.final_size, self.final_size), interpolation=cv2.INTER_AREA)
        square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)

        return transforms.ToTensor()(square)

def calculate_normalization_factors(color_img_rgb):
    """
    Calculates normalization factors based on the extreme pixels in an image.
    Input: img_rgb (Tensor) of shape [3, H, W] normalized to [0, 1]
    Output: (brightness_factor, gamma, hue_factor, saturation_factor)
    """
    # 1. Convert to HSV
    rgb = np.array(color_img_rgb.convert("RGB")) # (3,H,W) in [0,1]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)               # true HSV in [0,1]
    h = hsv[..., 0] / 179.0
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0

    h_f = torch.from_numpy(h).flatten()
    s_f = torch.from_numpy(s).flatten()
    v_f = torch.from_numpy(v).flatten()

    # --- IDENTIFY PIXELS ---
    v_black = v_f.min().item()

    low_sat = s_f < 0.2
    v_white = v_f[low_sat].max().item() if low_sat.any() else v_f.max().item()

    if low_sat.any():
        vv = v_f[low_sat]
        gray_scores = (vv - 0.5).abs()
        v_gray = vv[gray_scores.argmin()].item()
    else:
        v_gray = v_f[(v_f - 0.5).abs().argmin()].item()

    sat_mask = s_f > 0.2
    if sat_mask.any():
        hh = h_f[sat_mask]
        ss = s_f[sat_mask]
        vv = v_f[sat_mask]
        dist_to_red = torch.minimum((hh - 0.0).abs(), (hh - 1.0).abs())
        score = (ss + vv) - dist_to_red
        idx = score.argmax()
        h_red = hh[idx].item()
        s_red = ss[idx].item()
    else:
        h_red, s_red = 0.0, 1.0

    # --- FACTORS ---
    brightness = 1.0 / v_white if v_white > 1e-6 else 1.0
    contrast = 1.0 / ((v_white - v_black) + 1e-6)

    vg = max(1e-6, min(0.999999, v_gray * brightness))
    gamma = math.log(0.5) / math.log(vg) if vg not in (0.0, 1.0) else 1.0

    hue_shift = -h_red
    if hue_shift < -0.5: hue_shift += 1.0
    if hue_shift >  0.5: hue_shift -= 1.0

    sat_scale = 1.0 / s_red if s_red > 1e-6 else 1.0

    return hue_shift, sat_scale, contrast, gamma, brightness


class ValidationParamTfm:
    def __init__(self, img_size: int = 224):
        self._img_size = img_size
        self.post = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    def __call__(self, img, color_img):
        masked_image = MaskTransform(self, img)
        to_pil = ToPILImage()
        img_pil = to_pil(masked_image)
        image = img_pil
        hue_shift, sat_scale, contrast, gamma, brightness = calculate_normalization_factors(color_img)
        hue_shift = float(max(-0.5, min(0.5, hue_shift)))
        sat_scale = float(max(0.1, min(3.0, sat_scale)))
        contrast  = float(max(0.1, min(3.0, contrast)))
        brightness= float(max(0.1, min(3.0, brightness)))
        gamma     = float(max(0.1, min(3.0, gamma)))
        img = TF.adjust_hue(img, hue_shift)
        img = TF.adjust_saturation(img, sat_scale)
        img = TF.adjust_contrast(img, contrast)
        img = TF.adjust_gamma(img, gamma=gamma, gain=1.0)
        img = TF.adjust_brightness(img, brightness)

        return self.post(img)

def validate_only(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_tfm = args.val_tfm
    val_tfm._img_size = args.img_size

    # Dataset
    val_ds = EventDataset(
        csv_path=args.val_csv,
        img_dir=args.val_img_dir,
        event_col=args.event_col,
        img_col=args.img_col,
        target_cols=(args.spei30_col, args.spei1y_col, args.spei2y_col),
        tfm=args.val_tfm,
        k_max=args.k_max,
        train_mode=False,
        seed=args.seed,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # Load model (make sure to get rid of pretrained = True)
    model = EventConvNeXtRegressor("convnext_small").to(device)
    model.load_state_dict(torch.load(args.convnext_model_path, map_location=device))
    model.eval()

    va_loss = 0.0
    va_preds, va_gts = [], []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for step, (x, mask, y) in enumerate(pbar):
            x = x.to(device, non_blocking=True)           # [B, K, 3, H, W]
            mask = mask.to(device, non_blocking=True)    # [B, K]
            y = y.to(device, non_blocking=True)

            # Model forward
            mu, sigma = model(x, mask)  # Adjust your model forward to handle mask shape [B, K]

            # Compute Gaussian NLL loss
            loss = gaussian_nll(y, mu, sigma)
            va_loss += float(loss.item())

            # Collect predictions for R² evaluation
            va_preds.append(mu.detach().cpu().numpy())
            va_gts.append(y.detach().cpu().numpy())

            pbar.set_postfix({"loss": va_loss / (step + 1)})

    # Concatenate all predictions/ground truths
    va_preds = np.concatenate(va_preds, axis=0)
    va_gts = np.concatenate(va_gts, axis=0)

    # Compute R² metrics
    v30, v1y, v2y = evaluate_spei_r2(va_gts, va_preds)
    avg = (v30 + v1y + v2y) / 3.0

    print(f"Validation | val_loss={va_loss/len(val_loader):.4f} "
          f"val_r2=({v30:.3f},{v1y:.3f},{v2y:.3f}) avg={avg:.3f}")

    return va_preds, va_gts

img_size = 512

class Args:
    ## dataset
    val_csv = "hackdata/sentinel-beetles/public_release/val.csv"  # placeholder for the evaluation .csv and images
    val_img_dir = "validation_images"

    ## .csv columns
    img_col = "relative_img_loc"
    event_col = "eventID"
    spei30_col = "SPEI_30d"
    spei1y_col = "SPEI_1y"
    spei2y_col = "SPEI_2y"

    ## transform
    val_tfm = MaskTransform(model_path="sam2.1_l.pt", final_size=img_size)

    ## dataloader settings
    batch_size = 1
    img_size = img_size
    k_max = 3
    seed = 42
    num_workers = 4

    ## which model to use
    convnext_model_path = "event_convnext_small_img_aug_best.pth"

class Model:
    def __init__(self):
        # best model weights
        self.checkpoint = "event_convnext_small_img_aug_best.pth"
        self.SAMpath = "sam2.1_l.pt"
        
        ## .csv columns
        self.img_col = "relative_img_loc"
        self.event_col = "eventID"
        self.spei30_col = "SPEI_30d"
        self.spei1y_col = "SPEI_1y"
        self.spei2y_col = "SPEI_2y"

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.k_max = k_max

        self.img_size = 224
        self.k_max = 4

    def load(input_dict):
        SAM(SAMpath)
        self.model = EventConvNeXtRegressor("convnext_small").to(device)
        self.model.eval()
        self.tfm = InterferenceTfm(img_size)
        self.model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    @torch.no_grad()
    def predict(self, intput_dict):
        # unpack the dictionary
        if self.model is None or self.tfm is None of self.device is None:
            raise RuntimeError("Failed!")

        imgs: List[Image.Image] = []
        color_imgs: List[Image.Image] = []
        for rec in input_dict:
            img = rec.get("relative_img", None)
            color_img = rec.get("colorpicker_img", None)
            if img = None:
                continue
            if color_img = None:
                continue
            imgs.append(img)
            color_imgs.append(color_img)

        imgs = imgs[: self.k_max]

        xd = [self.tfm(img[i], color_img[i]) for i in range(len(imgs))]
        x = torch.stack(xs, dim = 0)

        n, C, H, W = x.shape
        if n < self.k_max:
            pad = torch.zeros((self.k_max - n, C, H, W), dtype = x.dtype)
            x = torch.cat([x, pad], dim = 0)
            mask = torch.tensor([1.0] * n + [0.0] * (self.k_max - n), dtype = torch.float32)

        else:
            mask = torch.ones(self.k_max, dtype = torch.float32)

        x = x.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)

        mu, sigma = self.model(x, mask)
        mu = mu.unsqueeze(0).to(self.device)
        sigma = sigma.unsqueeze(0).to(self.device)

        return {"SPEI_30d": {"mu": float(mu[0]), "sigma": float(sigma[0])},
                "SPEI_1y": {"mu": float(mu[1]), "sigma": float(sigma[1])},
                "SPEI_2y": {"mu": float(mu[2]), "sigma": float(sigma[2])}}
