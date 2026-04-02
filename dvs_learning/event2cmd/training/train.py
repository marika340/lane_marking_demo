import os, sys
from os.path import join as opj
import time
from datetime import datetime
import itertools
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import importlib.util

# ==== your repo imports ====
from dataloading import dataloader  # must yield (event_seq, seg_gt, cmd_gt)

BASE = os.path.dirname(os.path.abspath(__file__))
EV2SEG_DIR = os.path.join(BASE, "..", "models_event2seg")
SEG2CMD_DIR = os.path.join(BASE, "..", "models_seg2cmd")

# Make local module helpers (e.g., ViTsubmodules.py) importable
sys.path.insert(0, os.path.abspath(EV2SEG_DIR))
sys.path.insert(0, os.path.abspath(SEG2CMD_DIR))

def load_module(mod_name, file_path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Load model libraries with distinct module names
model_library_event2seg = load_module("model_event2seg", os.path.join(EV2SEG_DIR, "model.py"))
model_library_seg2cmd  = load_module("model_seg2cmd",  os.path.join(SEG2CMD_DIR, "model.py"))

# sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../models_event2seg'))
# import model as model_library_event2seg
# sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../models_seg2cmd'))
# import model as model_library_seg2cmd

# ---------------------------
# Utils
# ---------------------------
def parse_dirs(basedir, spec):
    dirs = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            for i in range(int(a), int(b) + 1):
                dirs.append(opj(basedir, f"dataset{i}"))
        else:
            dirs.append(opj(basedir, f"dataset{part}"))
    return dirs

def make_workspace(base_dir, logdir):
    timestamp = datetime.now().strftime('d%m_%d_t%H_%M')
    base = opj(base_dir, logdir, f"joint_{timestamp}")
    ws = base
    k = 1
    while os.path.exists(ws):
        ws = f"{base}_{k}"
        k += 1
    os.makedirs(ws)
    return ws

# ---------------------------
# Joint Trainer
# ---------------------------
class JOINT_TRAINER:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # --- workspace + writer ---
        self.workspace = make_workspace(args.basedir, args.logdir)
        self.writer = SummaryWriter(self.workspace)
        with open(opj(self.workspace, 'args.txt'), 'w') as f:
            for k, v in sorted(vars(args).items()):
                f.write(f"{k} = {v}\n")

        # --- data ---
        train_dirs = parse_dirs(args.basedir, args.train_dir)
        val_dirs   = parse_dirs(args.basedir, args.val_dir)

        # Expect: train_loader yields (event_seq, seg_gt, cmd_gt)
        # Shapes:
        #   event_seq: (B,T,1,480,640) float in [-1,1] or similar
        #   seg_gt:    (B,T,1,480,640) or (B,T,1,60,90) depending on dataloader
        #   cmd_gt:    (B,T,2) or (B,2) depending on your choice
        self.train_loader, self.val_loader, (self.cmd_mean, self.cmd_std) = dataloader(
            data_dir=args.basedir,
            train_val_dirs=(train_dirs, val_dirs),
            seed=42,
            # If your dataloader needs hints, add flags like: joint=True, return_cmd=True, return_seg=True
        )

        with open(opj(self.workspace, "norm_stats.txt"), "w") as f:
            f.write("cmd_mean: " + " ".join(map(str, self.cmd_mean.tolist())) + "\n")
            f.write("cmd_std: " + " ".join(map(str, self.cmd_std.tolist())) + "\n")
            print("[INFO] Saved cmd_mean and cmd_std")

        def _as_torch_1d(x, device):
            # handles numpy arrays, lists, tensors
            if isinstance(x, torch.Tensor):
                return x.to(device=device, dtype=torch.float32)
            return torch.as_tensor(x, dtype=torch.float32, device=device)

        self.cmd_mean = _as_torch_1d(self.cmd_mean, self.device)
        self.cmd_std  = _as_torch_1d(self.cmd_std,  self.device)

        # --- models ---
        # D: ev -> seg (1 channel mask)
        self.D = getattr(model_library_event2seg, args.D_type)(
            output_dim=1, return_sequence=args.D_return_sequence
        ).to(self.device).float()

        # V: seg -> cmd (2-dim controls per frame or last frame)
        self.V = getattr(model_library_seg2cmd, args.V_type)(
            output_dim=2, return_sequence=args.V_return_sequence
        ).to(self.device).float()

        # --- optimizer & sched ---
        params = itertools.chain(self.D.parameters(), self.V.parameters())
        self.optimizer = torch.optim.Adam(params, lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=1e-5
        )

        self.global_step = 0

    # ---------- helpers ----------
    def _prep_BT_or_seq(self, x, target_hw=(60, 90), seq_expected=False):
        """
        - If seq_expected=False (e.g., your ViT that flattens), return (B*T, C, H, W).
        - If seq_expected=True (LSTM/sequence models), return (B, T, C, H, W).
        Always resize spatial dims to target_hw.
        """
        if x.dim() == 5:  # (B,T,C,H,W)
            B, T, C, H, W = x.shape
            if seq_expected:
                x = x.view(B*T, C, H, W)
                x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
                x = x.view(B, T, C, *target_hw)
                return x
            else:
                x = x.view(B*T, C, H, W)
                x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
                return x
        elif x.dim() == 4:  # already (B*T,C,H,W)
            return F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
        else:
            raise ValueError(f"Unexpected tensor rank {x.dim()}")

    def _maybe_soft(self, seg_logits):
        """
        Keep differentiability. If D already outputs a [0,1] mask, use as-is.
        If it outputs logits, pass through sigmoid.
        """
        with torch.no_grad():
            vmin = seg_logits.min().item()
            vmax = seg_logits.max().item()
        if vmin < -0.5 and vmax > 1.5:
            return torch.sigmoid(seg_logits)
        return seg_logits  # already soft [0,1]

    def _align_cmd(self, cmd, as_BT, T=None):
        """Returns (B*T,2) if as_BT, else (B,T,2)."""
        if cmd.dim() == 3:  # (B,T,2)
            B, T0, D = cmd.shape
            if as_BT:
                return cmd.view(B*T0, D)
            return cmd
        elif cmd.dim() == 2:  # (B,2)
            if T is None:
                return cmd if not as_BT else cmd  # ambiguous; assume no temporal
            # tile to T
            B, D = cmd.shape
            cmd = cmd[:, None, :].expand(B, T, D)
            return cmd.view(B*T, D) if as_BT else cmd
        else:
            raise ValueError(f"Unexpected cmd rank {cmd.dim()}")

    # ---------- training ----------
    def train(self):
        best_val = float('inf')
        for ep in range(self.args.epochs):
            self.D.train(); self.V.train()
            ep_l, ep_l_seg, ep_l_cmd = 0.0, 0.0, 0.0

            for batch in self.train_loader:
                # Expect: (event_seq, seg_gt, cmd_gt)
                event_seq, seg_gt, cmd_gt = batch
                event_seq = event_seq.to(self.device)
                seg_gt    = seg_gt.to(self.device)
                cmd_gt    = cmd_gt.to(self.device)

                # --- resize/shape for models ---
                # D: ev -> seg, V: seg -> cmd
                # D expects BT or sequence per args
                # D_seq = True : (B,T,C,H,W)->(B,T,1,h,w)
                # D_seq = False: (B,T,C,H,W)->(B*T,1,h,w)
                # V_seq = True : (B,T,1,h,w)->(B,T,2)
                # V_seq = False: (B*T,1,h,w)->(B*T,2)
                D_seq = self.args.D_return_sequence
                V_seq = self.args.V_return_sequence

                # Prepare events for D
                x_D = self._prep_BT_or_seq(event_seq, target_hw=(60, 90), seq_expected=D_seq)
                X_D = [x_D]  # your models take list inputs

                # Forward D: ev -> seg_pred (soft)
                seg_pred, _ = self.D(X_D)  # (B*T,1,60,90) if not seq; OR (B,T,1,60,90) if seq
                # Match seg_gt shape to seg_pred and resize if necessary
                if seg_pred.dim() == 4:   # (BT,1,H,W)
                    seg_gt_reshaped = self._prep_BT_or_seq(seg_gt, target_hw=seg_pred.shape[-2:], seq_expected=False)
                else:                      # (B,T,1,H,W)
                    seg_gt_reshaped = self._prep_BT_or_seq(seg_gt, target_hw=seg_pred.shape[-2:], seq_expected=True)

                # SEG loss (keep it soft; no threshold)
                L_seg = F.mse_loss(seg_pred, seg_gt_reshaped)

                # Prepare seg for V (soft, differentiable)
                seg_soft = self._maybe_soft(seg_pred)

                # If you want V to see BT or sequence:
                if seg_soft.dim() == 4 and V_seq:  # (B*T,1,H,W) -> (B,T,1,H,W)
                    B, T = event_seq.shape[:2]
                    seg_soft = seg_soft.view(B, T, *seg_soft.shape[1:])

                # Forward V: seg -> cmd
                if V_seq:
                    X_V = [seg_soft]              # (B,T,1,60,90)
                    cmd_pred, _ = self.V(X_V)     # (B,T,2)
                    cmd_gt_aligned = self._align_cmd(cmd_gt, as_BT=False)
                else:
                    if seg_soft.dim() == 5:
                        B, T, C, H, W = seg_soft.shape
                        seg_soft = seg_soft.view(B*T, C, H, W)
                    X_V = [seg_soft]              # (B*T,1,60,90)
                    cmd_pred, _ = self.V(X_V)     # (B*T,2)
                    cmd_gt_aligned = self._align_cmd(cmd_gt, as_BT=True, T=event_seq.shape[1])

                # CMD loss (use normalized or raw? keep training in normalized scale)
                L_cmd = F.mse_loss(cmd_pred, cmd_gt_aligned)

                # Total loss
                loss = self.args.lambda_seg * L_seg + self.args.lambda_cmd * L_cmd

                # Optional: stop command-gradient from flowing into D
                if self.args.stop_grad_into_D:
                    # recompute cmd path with detached seg (no grad into D)
                    self.optimizer.zero_grad(set_to_none=True)
                    seg_soft_det = seg_soft.detach()
                    X_V_det = [seg_soft_det]
                    cmd_pred_det, _ = self.V(X_V_det)
                    L_cmd_det = F.mse_loss(cmd_pred_det, cmd_gt_aligned)
                    loss = self.args.lambda_seg * L_seg + self.args.lambda_cmd * L_cmd_det
                else:
                    self.optimizer.zero_grad(set_to_none=True)

                loss.backward()

                # GradNorm calculation
                total_norm_sq = 0.0
                for p in itertools.chain(self.D.parameters(), self.V.parameters()):
                    if p.grad is not None:
                        param_norm_sq = p.grad.data.norm(2)
                        total_norm_sq += param_norm_sq.item() ** 2
                total_norm = total_norm_sq ** 0.5

                if self.global_step % 50 == 0:
                    self.writer.add_scalar("Loss/train", loss.item(), self.global_step)
                    self.writer.add_scalar("LossSeg/train", L_seg.item(), self.global_step)
                    self.writer.add_scalar("LossCmd/train", L_cmd.item(), self.global_step)
                    self.writer.add_scalar("GradNorm/train", total_norm, self.global_step)

                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        itertools.chain(self.D.parameters(), self.V.parameters()),
                        self.args.grad_clip
                    )
                
                self.optimizer.step()

                # logging
                ep_l += loss.item()
                ep_l_seg += L_seg.item()
                ep_l_cmd += L_cmd.item()
                self.global_step += 1
                
            # ---- validation ----
            val_loss, val_seg, val_cmd = self.validate()
            lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()

            curr_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("LearningRate", curr_lr, ep)
            self.writer.add_scalar("Loss/val", val_loss, ep)
            self.writer.add_scalar("LossSeg/val", val_seg, ep)
            self.writer.add_scalar("LossCmd/val", val_cmd, ep)

            N = len(self.train_loader)
            print(f"[Epoch {ep+1:03d}] "
                  f"train: L={ep_l/N:.4f} (seg {ep_l_seg/N:.4f}, cmd {ep_l_cmd/N:.4f}) | "
                  f"val: L={val_loss:.4f} (seg {val_seg:.4f}, cmd {val_cmd:.4f}) | lr={lr:.6f}")

            # save checkpoints (together)
            ckpt = {
                "D_state": self.D.state_dict(),
                "V_state": self.V.state_dict(),
                "opt": self.optimizer.state_dict(),
                "epoch": ep+1,
                "args": vars(self.args),
            }
            torch.save(ckpt, opj(self.workspace, f"joint_{ep+1:03d}.pth"))
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ckpt, opj(self.workspace, f"best_joint_{ep+1:03d}.pth"))

    @torch.no_grad()
    def validate(self):
        self.D.eval(); self.V.eval()
        total, total_seg, total_cmd, n = 0.0, 0.0, 0.0, 0
        for batch in self.val_loader:
            event_seq, seg_gt, cmd_gt = batch
            event_seq = event_seq.to(self.device)
            seg_gt    = seg_gt.to(self.device)
            cmd_gt    = cmd_gt.to(self.device)

            # D forward
            x_D = self._prep_BT_or_seq(event_seq, target_hw=(60,90), seq_expected=self.args.D_return_sequence)
            seg_pred, _ = self.D([x_D])

            # seg loss
            if seg_pred.dim() == 4:
                seg_gt_r = self._prep_BT_or_seq(seg_gt, target_hw=seg_pred.shape[-2:], seq_expected=False)
            else:
                seg_gt_r = self._prep_BT_or_seq(seg_gt, target_hw=seg_pred.shape[-2:], seq_expected=True)
            L_seg = F.mse_loss(seg_pred, seg_gt_r)

            # V forward (use soft segs from D)
            seg_soft = self._maybe_soft(seg_pred)
            if seg_soft.dim() == 4 and self.args.V_return_sequence:
                B, T = event_seq.shape[:2]
                seg_soft = seg_soft.view(B, T, *seg_soft.shape[1:])

            if self.args.V_return_sequence:
                cmd_pred, _ = self.V([seg_soft])         # (B,T,2)
                cmd_gt_a = self._align_cmd(cmd_gt, as_BT=False)
            else:
                if seg_soft.dim() == 5:
                    B, T, C, H, W = seg_soft.shape
                    seg_soft = seg_soft.view(B*T, C, H, W)
                cmd_pred, _ = self.V([seg_soft])         # (B*T,2)
                cmd_gt_a = self._align_cmd(cmd_gt, as_BT=True, T=event_seq.shape[1])

            L_cmd = F.mse_loss(cmd_pred, cmd_gt_a)
            L = self.args.lambda_seg * L_seg + self.args.lambda_cmd * L_cmd

            total += L.item()
            total_seg += L_seg.item()
            total_cmd += L_cmd.item()
            n += 1

        return (total/n, total_seg/n, total_cmd/n)

# ---------------------------
# Argparse
# ---------------------------
def argparsing(filename=None):
    import configargparse, getpass
    uname = getpass.getuser()

    if filename is not None:
        default_config_files = [filename]
    else:
        default_config_files = [f'/home/{uname}/catkin_ws/src/lane_marking_demo/dvs_learning/event2cmd/training/configs/train_config_event2cmd.txt']
    
    p = configargparse.ArgumentParser(default_config_files=default_config_files)
    p.add_argument('--config', is_config_file=True, help='Path to config file', default=default_config_files[0])
    p.add_argument('--basedir', type=str, default=f'/home/{uname}/catkin_ws/src/lane_marking_demo')
    p.add_argument('--logdir', type=str, default='logs')
    p.add_argument('--device', type=str, default='cuda')

    # data
    p.add_argument('--train_dir', type=str, default='417-500,617-700')
    p.add_argument('--val_dir', type=str, default='501-516,701-716')

    # models
    p.add_argument('--D_type', type=str, default='ViT')       # ev->seg
    p.add_argument('--V_type', type=str, default='ViT')       # seg->cmd
    p.add_argument('--D_return_sequence', action='store_true', default=False)
    p.add_argument('--V_return_sequence', action='store_true', default=False)

    # opt
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--grad_clip', type=float, default=1.0)

    # loss weights
    p.add_argument('--lambda_seg', type=float, default=1.0)
    p.add_argument('--lambda_cmd', type=float, default=1.0)

    # options
    p.add_argument('--stop_grad_into_D', action='store_true', help='Do not backprop L_cmd into D')
    return p.parse_args()

if __name__ == '__main__':
    args = argparsing()
    trainer = JOINT_TRAINER(args)
    trainer.train()
