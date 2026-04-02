import os, sys
from os.path import join as opj
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# --- import your code ---
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
sys.path.append(opj(HERE, "../models"))

from train import argparsing, parse_dirs
from dataloading import dataloader
import model as model_library
import re

def get_next_prefix(out_dir="pred_debug", base="val"):
    os.makedirs(out_dir, exist_ok=True)
    existing = os.listdir(out_dir)

    # match val12_pred.png OR val12_gt.png OR val12_event.png
    pattern = re.compile(
        rf"^{re.escape(base)}(\d+)_(pred|gt|event)\.png$"
    )

    max_idx = -1
    for name in existing:
        m = pattern.match(name)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx

    return f"{base}{max_idx + 1}"


@torch.no_grad()
def build_model(model_type: str, device: torch.device):
    """
    Mirror TRAINER.__init__ model construction so shapes match.
    """
    if model_type == "SimpleCNN":
        model = getattr(model_library, model_type)(
            output_dim=2,
            return_sequence=False
        )
    elif model_type == "ViT":
        model = getattr(model_library, model_type)(
            output_dim=2,
            return_sequence=True
        )
    else:
        # fallback for sequence-y models like LSTMNetVIT
        model = getattr(model_library, model_type)(
            output_dim=2,
            return_sequence=True
        )

    model = model.to(device).float().eval()
    return model


@torch.no_grad()
def load_weights(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    """
    Load a state_dict that was saved with torch.save(model.state_dict()).
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.no_grad()
def prep_batch_for_model(event_batch, mask_batch, model_type, device):
    """
    Prepare one batch exactly like training/validate does for THAT model_type.

    Returns:
        X_for_model          -> list[...] to feed directly into model(...)
        mask_proc_for_loss   -> GT mask downsampled + thresholded for comparison
        event_for_vis_small  -> event frames (downsampled for plotting ONLY)
        meta_shape_info      -> dict to help pick_frame_for_visual() slice correctly
    """
    event_batch = event_batch.to(device)
    mask_batch  = mask_batch.to(device)

    # -------------------------------------------------------------------------
    # CASE 1: SimpleCNN
    # -------------------------------------------------------------------------
    if model_type == "SimpleCNN":
        # event_batch: (B,1,H,W)
        # resize input to (60,90) before model
        event_small = F.interpolate(
            event_batch, size=(60, 90),
            mode='bilinear', align_corners=False
        )  # (B,1,60,90)

        mask_small = F.interpolate(
            mask_batch, size=(60, 90),
            mode='bilinear', align_corners=False
        )
        mask_small = (mask_small > 0.08).float()  # (B,1,60,90)

        X_for_model = [event_small]  # model expects [event_small]

        meta = {
            "type": "SimpleCNN",
            "B": event_small.shape[0],
        }

        # for visualization we can just reuse event_small
        return X_for_model, mask_small, event_small, meta

    # -------------------------------------------------------------------------
    # CASE 2: ViT
    # -------------------------------------------------------------------------
    if model_type == "ViT":
        # TRAINER.validate branch for ViT:
        #   B,T,C,H,W = event_batch.shape
        #   event_batch_flat = event_batch.view(B*T,C,H,W)   # NO RESIZE for input
        #   mask_batch_flat  = mask_batch.view(B*T,C,H,W)
        #   mask_small = F.interpolate(mask_batch_flat, (60,90)), threshold
        #   X = [event_batch_flat]
        B, T, C, H, W = event_batch.shape

        event_btchw = event_batch.view(B * T, C, H, W)  # (B*T,1,480,640) for model
        mask_btchw  = mask_batch.view(B * T, C, H, W)

        mask_small  = F.interpolate(
            mask_btchw, size=(60, 90),
            mode='bilinear', align_corners=False
        )  # (B*T,1,60,90)
        mask_small  = (mask_small > 0.08).float()

        X_for_model = [event_btchw]  # feed full-res frames directly

        # for visualization, let's ALSO prepare a downsampled event version
        event_vis_small = F.interpolate(
            event_btchw, size=(60, 90),
            mode='bilinear', align_corners=False
        )  # (B*T,1,60,90) -> nice to display next to pred/mask

        meta = {
            "type": "ViT",
            "B": B,
            "T": T,
        }

        return X_for_model, mask_small, event_vis_small, meta

    # -------------------------------------------------------------------------
    # CASE 3: sequence models like LSTMNetVIT (the "else" branch in TRAINER)
    # -------------------------------------------------------------------------
    # Sequential models expect (B,T,1,60,90)
    B, T, C, H, W = event_batch.shape

    ev_btchw = event_batch.view(B * T, C, H, W)
    ev_btchw_small = F.interpolate(
        ev_btchw, size=(60, 90),
        mode='bilinear', align_corners=False
    )
    event_seq_small = ev_btchw_small.view(B, T, C, 60, 90)  # (B,T,1,60,90)

    ms_btchw = mask_batch.view(B * T, C, H, W)
    ms_btchw_small = F.interpolate(
        ms_btchw, size=(60, 90),
        mode='bilinear', align_corners=False
    )
    mask_seq_small = ms_btchw_small.view(B, T, C, 60, 90)
    mask_seq_small = (mask_seq_small > 0.08).float()

    X_for_model = [event_seq_small]  # model expects [B,T,1,60,90]

    meta = {
        "type": "SEQ",
        "B": B,
        "T": T,
    }

    # for visualization, we'll just reuse event_seq_small
    return X_for_model, mask_seq_small, event_seq_small, meta


@torch.no_grad()
def forward_model(model, X_for_model):
    """
    All your models return (pred, extras). We only need pred.
    """
    pred, extras = model(X_for_model)
    return pred  # shape depends on model_type branch


def _to_np_img(t: torch.Tensor):
    """
    Convert (1,H,W) or (H,W) tensor -> numpy [0,1] for plotting.
    If values look like raw logits, we'll run sigmoid.
    """
    if t.dim() == 3 and t.size(0) == 1:
        t = t[0]  # (H,W)
    if (t.min() < 0.0) or (t.max() > 1.0):
        t = torch.sigmoid(t)
    t = t.detach().cpu().float().clamp(0,1).numpy()
    return t


def pick_frame_for_visual(pred, event_vis_small, mask_small, meta):
    """
    Return (ev0, pr0, gt0) each ~ (1,H,W) for plotting and saving.

    Shapes:
    - SimpleCNN branch:
        event_vis_small: (B,1,H,W)
        mask_small:      (B,1,H,W)
        pred:            (B,1,H,W)

    - ViT branch:
        event_vis_small: (B*T,1,H,W)         (H,W ~ 60x90 for vis)
        mask_small:      (B*T,1,H,W)         (60x90 thresholded GT)
        pred:            (B*T,1,H,W) ? (what model outputs)
      We choose frame_idx = 0 (batch0, time0), which is index 0.

    - SEQ branch:
        event_vis_small: (B,T,1,H,W)
        mask_small:      (B,T,1,H,W)
        pred: could be
            (B,T,1,H,W) OR
            (B,1,H,W)   OR
            (B*T,1,H,W)
    """
    tp = meta["type"]

    # -------- SimpleCNN --------
    if tp == "SimpleCNN":
        ev0 = event_vis_small[0]  # (1,H,W)
        gt0 = mask_small[0]       # (1,H,W)
        pr0 = pred[0]             # (1,H,W)
        return ev0, pr0, gt0

    # -------- ViT --------
    if tp == "ViT":
        # event_vis_small: (B*T,1,60,90)
        # mask_small:      (B*T,1,60,90)
        # pred:            likely (B*T,1,60,90)
        ev0 = event_vis_small[0]
        gt0 = mask_small[0]

        if pred.dim() == 4:
            # (B*T,1,H,W)  or  (B,1,H,W) depending on arch
            pr0 = pred[0]
        else:
            raise RuntimeError(f"[viz_eval] Unexpected pred shape for ViT: {pred.shape}")

        return ev0, pr0, gt0

    # -------- SEQ (LSTMNetVIT style) --------
    B = meta["B"]
    T = meta["T"]

    # Take batch0, time0
    ev0 = event_vis_small[0,0]  # (1,H,W)
    gt0 = mask_small[0,0]       # (1,H,W)

    if pred.dim() == 5 and pred.shape[:2] == (B, T):
        # pred is (B,T,1,H,W)
        pr0 = pred[0,0]
    elif pred.dim() == 4 and pred.shape[0] == B:
        # pred is (B,1,H,W) broadcast in time
        pr0 = pred[0]
    elif pred.dim() == 4 and pred.shape[0] == (B * T):
        # pred is (B*T,1,H,W)
        pr0 = pred[0]
    else:
        raise RuntimeError(f"[viz_eval] Don't know how to slice pred with shape {pred.shape}")

    return ev0, pr0, gt0


def visualize_and_save(ev0, pr0, gt0, out_dir="pred_debug", prefix="sample0"):
    os.makedirs(out_dir, exist_ok=True)

    ev_img = _to_np_img(ev0)
    pr_img = _to_np_img(pr0)
    gt_img = _to_np_img(gt0)

    # inline figure
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("input event")
    plt.imshow(ev_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("prediction")
    plt.imshow(pr_img, cmap='viridis')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("ground truth")
    plt.imshow(gt_img, cmap='viridis')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # save raw tensors too (handy for Overleaf figures etc.)
    vutils.save_image(torch.sigmoid(pr0), opj(out_dir, f"{prefix}_pred.png"))
    vutils.save_image(gt0.float(),        opj(out_dir, f"{prefix}_gt.png"))
    vutils.save_image(ev0.float(),        opj(out_dir, f"{prefix}_event.png"))

    print(f"[viz_eval] wrote {out_dir}/{prefix}_*.png")


def main():
    # parse same config file you used for training
    args = argparsing()

    # select device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # rebuild val loader
    train_dirs = parse_dirs(args.basedir, args.train_dir)
    val_dirs   = parse_dirs(args.basedir, args.val_dir)

    _, val_loader = dataloader(
        data_dir=args.basedir,
        train_val_dirs=(train_dirs, val_dirs),
        seed=42
    )

    # grab checkpoint path from config / cmdline
    ckpt_path = args.checkpoint_path[0]
    print(f"[viz_eval] using checkpoint: {ckpt_path}")

    # build + load model
    model = build_model(args.model_type, device)
    model = load_weights(model, ckpt_path, device)

    # pull one batch from val
    event_batch, mask_batch = next(iter(val_loader))

    # preprocess for the right model_type ("SimpleCNN", "ViT", or sequence)
    X_for_model, mask_proc, event_for_vis_small, meta = prep_batch_for_model(
        event_batch, mask_batch, args.model_type, device
    )

    # run model
    pred = forward_model(model, X_for_model)

    # choose batch0/frame0 for visualization
    ev0, pr0, gt0 = pick_frame_for_visual(pred, event_for_vis_small, mask_proc, meta)

    prefix = get_next_prefix(out_dir="pred_debug", base="val")

    visualize_and_save(ev0, pr0, gt0, out_dir="pred_debug", prefix=prefix)

    # show + save
    visualize_and_save(ev0, pr0, gt0, out_dir="pred_debug", prefix="val0")


if __name__ == "__main__":
    main()
