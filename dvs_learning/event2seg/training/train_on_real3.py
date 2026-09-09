import os, sys
import glob
from os.path import join as opj
import numpy as np
import torch
import time
from datetime import datetime
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataloading_real import *
sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../models'))
import model_on_real as model_library

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import getpass
uname = getpass.getuser()


def parse_dirs(basedir, spec):
    """Turn '132-135,140,150-151' into full dataset paths."""
    dirs = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            for i in range(int(start), int(end) + 1):
                dirs.append(opj(basedir, f"dataset{i}"))
        else:
            dirs.append(opj(basedir, f"dataset{part}"))
    return dirs

def make_workspace(base_dir, logdir):
    timestamp = datetime.now().strftime('d%m_%d_t%H_%M')
    base_workspace = opj(base_dir, logdir, timestamp)
    workspace = base_workspace
    suffix = 1
    while os.path.exists(workspace):
        workspace = f"{base_workspace}_{suffix}"
        suffix += 1
    os.makedirs(workspace)
    return workspace


class TRAINER:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        self.basedir = args.basedir
        # self.dataset_name = args.dataset
        # self.dataset_path = opj(self.basedir, self.dataset_name)
        self.event_npy_path = args.event_npy_path
        self.align_mode = args.align_mode
        self.model_type = args.model_type
        self.epochs = args.epochs
        self.lr = args.lr
        self.logdir = args.logdir
        self.combine_checkpoints = args.combine_checkpoints
        self.previous_tag = None

        expname = datetime.now().strftime('d%m_%d_t%H_%M')
        self.workspace = opj(self.basedir, self.logdir, expname)
        ctr = 2
        while os.path.exists(self.workspace):
            self.workspace = opj(self.basedir, self.logdir, f"{expname}_{ctr}")
            ctr += 1
        os.makedirs(self.workspace)
        self.writer = SummaryWriter(self.workspace)

        # save ordered args, config, and a logfile to write stdout to
        if self.args is not None:
            f = opj(self.workspace, 'args.txt')
            with open(f, 'w') as file:
                for arg in sorted(vars(self.args)):
                    attr = getattr(self.args, arg)
                    file.write('{} = {}\n'.format(arg, attr))
            f = opj(self.workspace, 'config.txt')
            # if self.args.config is not None:
            #     file.write(open(self.args.config, 'r').read())

            with open(self.args.config, 'r') as f:
                config_contents = f.read()

            with open(opj(self.workspace, "config.txt"), 'w') as file:
                file.write(config_contents)


            # with open(f, 'w') as file:
            #     file.write(open(self.args.config, 'r').read())
        f = opj(self.workspace, 'log.txt')
        self.logfile = open(f, 'w')

        print(f"[SETUP] Workspace: {self.workspace}")

        # Load dataset
        print("[INFO] Loading dataset...")
        print("[DEBUG] args.event_npy_path =", args.event_npy_path)
        print("[DEBUG] args.align_mode =", args.align_mode)
        
        # Parse explicit train/val dirs from config
        train_loader, val_loader = dataloader(
            data_dir=args.real_data_root,
            event_npy_path=args.event_npy_path,
            align_mode=args.align_mode,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            seed=42
        )

        # collate = collate_fn if self.model_type == "SimpleCNN" else None

        self.train_data = train_loader
        self.val_data = val_loader

        # ---- fetch normalization stats from the underlying DrivingDataset ----
        # def _unwrap_base(ds):
        #     # ds can be Subset(ConcatDataset([...])) or Subset(DrivingDataset) or DrivingDataset
        #     from torch.utils.data import Subset, ConcatDataset
        #     base = ds
        #     if isinstance(base, Subset):
        #         base = base.dataset
        #     if isinstance(base, ConcatDataset):
        #         # pick first component dataset (they all compute stats the same way)
        #         base = base.datasets[0]
        #     return base
        
        # base_ds = _unwrap_base(self.train_data.dataset)
        # self.cmd_mean = base_ds.cmd_mean.to(self.device)
        # self.cmd_std  = base_ds.cmd_std.to(self.device)
        # print("[INFO] cmd_mean:", self.cmd_mean.tolist())
        # print("[INFO] cmd_std:",  self.cmd_std.tolist())

        print(f"[INFO] Initializing model: {self.model_type}")

        if self.model_type == "SimpleCNN":
            self.model = getattr(model_library, self.model_type)(
                output_dim=2, return_sequence=False
            ).to(self.device).float()
        elif self.model_type == "ViT":
            self.model = getattr(model_library, self.model_type)(
                output_dim=2, return_sequence=True
            ).to(self.device).float()
        else:
            self.model = getattr(model_library, self.model_type)(
                output_dim=2, return_sequence=True
            ).to(self.device).float()
        
        # ---- Load pretrained checkpoint if provided ----
        # self.finetune = True -> train from pretrained checkpoint; False -> train from scratch baseline
        self.finetune = args.checkpoint_path is not None and len(args.checkpoint_path) > 0

        if self.finetune:
            ckpt_path = args.checkpoint_path[0]
            self.mylogger(f"[LOADER] Fine-tuning from pretrained checkpoint: {ckpt_path}")
            self.load_from_checkpoint(ckpt_path)
        else:
            self.mylogger("[LOADER] No checkpoint provided. Training from scratch baseline.")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # BCE loss on logits
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0).to(self.device))
        # MSE loss
        # self.criterion = torch.nn.MSELoss()

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-5
        )

        self.total_its = 0

        # print(f"[INFO] Initializing model: {self.model_type}")
        # # self.model = getattr(model_library, self.model_type)().to(self.device).float()
        # # self.model = getattr(model_library, self.model_type)(output_dim=5, return_sequence=True).to(self.device).float()
        # # self.model = getattr(model_library, self.model_type)(output_dim=5, return_sequence=False).to(self.device).float()
        # # self.model = getattr(model_library, self.model_type)(output_dim=2, return_sequence=False).to(self.device).float()
        # self.model = getattr(model_library, self.model_type)(output_dim=2, return_sequence=True).to(self.device).float()
        # # args.return_sequence = False
        # # self.model = getattr(model_library, self.model_type)(
        # #     output_dim=5,
        # #     return_sequence=getattr(args, "return_sequence", True)
        # # ).to(self.device).float()
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # # loss calculation
        # self.criterion = torch.nn.BCEWithLogitsLoss()
        # # Scheduler
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, T_max=self.epochs, eta_min=1e-5
        # )

        # self.total_its = 0

    def train(self):
        if self.finetune:
            print(f"[TRAIN] Fine-tuning on real data for {self.epochs} epochs")
        else:
            print(f"[TRAIN] Training from scratch on real data for {self.epochs} epochs")

        best_val_loss = float('inf') # Track the best validation loss
            
        for ep in range(self.epochs):
            self.model.train()
            ep_loss = 0
            gradnorm = 0

            for event_batch, mask_batch in self.train_data:
                event_batch = event_batch.to(self.device)    # (B,T,1,480,640)
                mask_batch  = mask_batch.to(self.device)        # (B,T,1,480,640)

                # # DEBUG
                # # Inside training loop
                # print("Drive:", drive_batch[0])
                # print("Mean:", drive_batch.mean(), "STD:", drive_batch.std())
                # print("Mask max/min:", mask_batch.max(), mask_batch.min())
                # break  # Just once

                if self.model_type == "SimpleCNN":
                    # Already flattened by simplecnn_collate_fn:
                    # event_batch: (B,1,H,W), mask_batch: (B,1,H,W)
                    event_batch = F.interpolate(
                        event_batch, size=(60, 90),
                        mode='bilinear', align_corners=False
                    )
                    mask_batch = F.interpolate(
                        mask_batch, size=(60, 90),
                        mode='bilinear', align_corners=False
                    )
                    mask_batch = (mask_batch > 0.08).float()
                    X = [event_batch]
                    pred, _ = self.model(X)   # (B,5)
                    loss = self.criterion(pred, mask_batch)
                elif self.model_type == "ViT":
                    Be, Te, Ce, He, We = event_batch.shape
                    Bm, Tm, Cm, Hm, Wm = mask_batch.shape

                    event_batch = event_batch.view(Be * Te, Ce, He, We)
                    mask_batch = mask_batch.view(Bm * Tm, Cm, Hm, Wm)

                    event_batch = F.interpolate(
                        event_batch, size=(60, 90),
                        mode='bilinear', align_corners=False
                    )
                    mask_batch = F.interpolate(
                        mask_batch, size=(60, 90),
                        mode='bilinear', align_corners=False
                    )
                    # print(f"[DEBUG] train mask unique vals: {torch.unique(mask_batch)}")
                    mask_batch = (mask_batch > 0.08).float() # B*T, C, H, W
                    # print(f"[DEBUG] train mask after threshold unique vals: {torch.unique(mask_batch)}")
                    X = [event_batch]
                    pred, _ = self.model(X)   # (B*T, C, H, W)
                    # print(f"[DEBUG] pred shape: {pred.shape}, mask_batch shape: {mask_batch.shape}")
                    loss = self.criterion(pred, mask_batch)
                else:
                    # Sequential models (e.g. LSTMNetVIT) expect (B,T,1,H,W)
                    B, T, C, H, W = event_batch.shape
                    # print(f"[DEBUG] mask_batch shape before resize: {mask_batch.shape}")
                    event_batch = event_batch.view(B * T, C, H, W)
                    event_batch = F.interpolate(
                        event_batch, size=(60, 90),
                        mode='bilinear', align_corners=False
                    )
                    event_batch = event_batch.view(B, T, C, 60, 90)
                    print(f"[DEBUG] event_batch shape after resize: {event_batch.shape}")
                    mask_batch = mask_batch.view(B * T, C, H, W)
                    mask_batch = F.interpolate(
                        mask_batch, size=(60, 90),
                        mode='bilinear', align_corners=False
                    )
                    mask_batch = mask_batch.view(B, T, C, 60, 90)
                    print(f"[DEBUG] train mask unique vals: {torch.unique(mask_batch)}")
                    mask_batch = (mask_batch > 0.08).float()
                    print(f"[DEBUG] train mask after threshold unique vals: {torch.unique(mask_batch)}")
                    X = [event_batch]
                    pred, _ = self.model(X)   # (B,T,1,60,90)
                    loss = self.criterion(pred, mask_batch)
                    # # Resize input sequence from (B, T, 1, 480, 640) → (B, T, 1, 90, 60)
                    # B, T, C, H, W = mask_batch.shape
                    # mask_batch = mask_batch.view(B * T, C, H, W)
                    # mask_batch = F.interpolate(mask_batch, size=(90, 60), mode='bilinear', align_corners=False)
                    # mask_batch = mask_batch.view(B, T, C, 90, 60)
                # X = [mask_batch]
                # pred, _ = self.model(X)                 # pred.shape == (B,2)
                # # print(f'pred.shape = {pred.shape}')
                # # If the model is misconfigured to return last-step only, adapt:
                # if pred.dim() == 2 and pred.size(-1) == 5:
                #     # Expand to (B,1,5) to keep loss consistent, compare to last label
                #     pred = pred.unsqueeze(1)
                #     drive_last = drive_batch[:, -1:].contiguous()
                #     loss = F.mse_loss(pred, drive_last)
                # else:
                #     # Full sequence supervision
                #     assert pred.shape == drive_batch.shape, \
                #         f"pred {tuple(pred.shape)} vs target {tuple(drive_batch.shape)}"
                #     loss = F.mse_loss(pred, drive_batch)
                # # loss = F.mse_loss(pred, drive_batch)
                self.optimizer.zero_grad()
                loss.backward()

                # === Debug gradient magnitudes ===
                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         print(f"[GRAD] {name}: {param.grad.abs().mean().item():.6f}")
                # =================================

                # For GradNorm/train
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                gradnorm += total_norm ** 0.5

                self.optimizer.step()

                ep_loss += loss.item()
                self.total_its += 1

            # break
            avg_train_loss = ep_loss / len(self.train_data)
            gradnorm /= len(self.train_data)

            val_loss = self.validate()

            # Step the scheduler
            self.scheduler.step()

            # Get current LR (CosineAnnealingLR keeps it in optimizer.param_groups)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log to TensorBoard
            self.writer.add_scalar("LearningRate", current_lr, ep)

            print(f"[Epoch {ep+1}] Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
            self.writer.add_scalar('Loss/train', avg_train_loss, ep)
            self.writer.add_scalar('Loss/val', val_loss, ep)
            self.writer.add_scalar('GradNorm/train', gradnorm, ep)

            torch.save(self.model.state_dict(), opj(self.workspace, f'model_{ep+1:03d}.pth'))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), opj(self.workspace, f'best_model_{ep+1:03d}.pth'))
                print(f"[INFO] Best model updated at epoch {ep+1} with val loss {val_loss:.4f}")

    # def validate(self):
    #     self.model.eval()
    #     total_loss = 0.0
    #     num_batches = 0

    #     with torch.no_grad():
    #         # for batch in self.val_data:
    #         #     # Unpack exactly as in training
    #         #     event, des_vel, quat = batch
    #         #     event   = event.to(self.device)
    #         #     des_vel = des_vel.to(self.device)
    #         #     quat    = quat.to(self.device)

    #         #     # Build the input list for the model
    #         #     X = [event, des_vel, quat]
    #         #     pred, _ = self.model(X)                 # pred.shape == (B, 3)

    #         #     # Compute MSE loss against the same des_vel target
    #         #     loss = F.mse_loss(pred, des_vel)
    #         for mask_batch, drive_batch in self.val_data:
    #             mask_batch  = mask_batch.to(self.device)        # (B,1,480,640)
    #             drive_batch = drive_batch.to(self.device)    # (B,2)

    #             # Resize input sequence from (B, T, 1, 480, 640) → (B, T, 1, 60, 90)
    #             B, T, C, H, W = mask_batch.shape
    #             mask_batch = mask_batch.view(B * T, C, H, W)
    #             mask_batch = F.interpolate(mask_batch, size=(60, 90), mode='bilinear', align_corners=False)
    #             mask_batch = mask_batch.view(B, T, C, 60, 90)

    #             # # Resize input sequence from (B, T, 1, 480, 640) → (B, T, 1, 90, 60)
    #             # B, T, C, H, W = mask_batch.shape
    #             # mask_batch = mask_batch.view(B * T, C, H, W)
    #             # mask_batch = F.interpolate(mask_batch, size=(90, 60), mode='bilinear', align_corners=False)
    #             # mask_batch = mask_batch.view(B, T, C, 90, 60)

    #             X = [mask_batch]
    #             pred, _ = self.model(X)                 # pred.shape == (B,2)

    #             loss = F.mse_loss(pred, drive_batch)

    #             total_loss += loss.item()
    #             num_batches += 1

    #     # Restore train mode
    #     self.model.train()

    #     return (total_loss / num_batches) if num_batches > 0 else 0.0
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for event_batch, mask_batch in self.val_data:
                event_batch = event_batch.to(self.device)
                mask_batch  = mask_batch.to(self.device)

                if self.model_type == "SimpleCNN":
                    # Already flattened by simplecnn_collate_fn:
                    # event_batch: (B,1,H,W), mask_batch: (B,1,H,W)
                    event_batch = F.interpolate(
                        event_batch, size=(60, 90),
                        mode='bilinear', align_corners=False
                    )
                    mask_batch = F.interpolate(
                        mask_batch, size=(60, 90),
                        mode='bilinear', align_corners=False
                    )
                    print(f"[DEBUG] validate mask unique vals: {torch.unique(mask_batch)}")
                    mask_batch = (mask_batch > 0.08).float()
                    print(f"[DEBUG] validate mask after threshold unique vals: {torch.unique(mask_batch)}")
                    X = [event_batch]
                    pred, _ = self.model(X)   # (B,1,H,W)
                    loss = self.criterion(pred, mask_batch)
                elif self.model_type == "ViT":
                    Be, Te, Ce, He, We = event_batch.shape
                    Bm, Tm, Cm, Hm, Wm = mask_batch.shape

                    event_batch = event_batch.view(Be * Te, Ce, He, We)
                    mask_batch = mask_batch.view(Bm * Tm, Cm, Hm, Wm)

                    event_batch = F.interpolate(
                        event_batch, size=(60, 90),
                        mode='bilinear', align_corners=False
                    )
                    mask_batch = F.interpolate(
                        mask_batch, size=(60, 90),
                        mode='bilinear', align_corners=False
                    )
                    # mask_batch = mask_batch.view(B, C, 60, 90)
                    mask_batch = (mask_batch > 0.08).float()
                    X = [event_batch]
                    pred, _ = self.model(X)   # (B,1,H,W)
                    loss = self.criterion(pred, mask_batch)
                else:
                    # Sequential models (e.g. LSTMNetVIT) expect (B,T,1,H,W)
                    B, T, C, H, W = event_batch.shape
                    event_batch = event_batch.view(B * T, C, H, W)
                    event_batch = F.interpolate(
                        event_batch, size=(60, 90),
                        mode='bilinear', align_corners=False
                    )
                    event_batch = event_batch.view(B, T, C, 60, 90)
                    mask_batch = mask_batch.view(B * T, C, H, W)
                    mask_batch = F.interpolate(
                        mask_batch, size=(60, 90),
                        mode='bilinear', align_corners=False
                    )
                    mask_batch = mask_batch.view(B, T, C, 60, 90)
                    mask_batch = (mask_batch > 0.08).float()
                    X = [event_batch]
                    pred, _ = self.model(X)   # (B,T,1,60,90)
                    loss = self.criterion(pred, mask_batch)

                total_loss += loss.item()
                num_batches += 1

        self.model.train()
        return (total_loss / num_batches) if num_batches > 0 else 0.0


    # a useful logger that prints messages to stdout and writes them to a logfile
    # it chunks messages based on their tags at the beginning of each message
    def mylogger(self, msg):
        # Extract the tag from the message using square brackets
        tag = msg.split('[')[1].split(']')[0] if '[' in msg and ']' in msg else None

        # Check if there's a tag and if it's different from the previous one print a newline
        if tag is not None and tag != self.previous_tag:
            print('\n', end='') # Print a newline
            self.logfile.write('\n')

        print(msg)
        self.logfile.write(msg+'\n')

        self.previous_tag = tag
    
    def _infer_model_type_from_keys(self, state_dict_keys):
        keys = list(state_dict_keys)
        # Heuristics for your repo:
        if any(k.startswith('encoder_blocks.') or k.startswith('decoder') or k.startswith('down_sample') for k in keys):
            return 'ViT'
        if any(k.startswith('conv1') or k.startswith('conv2') or k.startswith('conv3') or k.startswith('fc1') for k in keys):
            return 'SimpleCNN'
        # Add more here if you have LSTMNetVIT etc.
        if any('.lstm.' in k for k in keys) or any('transformer' in k for k in keys):
            return 'LSTMNetVIT'
        return None

    def load_from_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        inferred = self._infer_model_type_from_keys(state_dict.keys())

        if inferred is not None and inferred != self.model_type:
            self.mylogger(f"[LOADER] Checkpoint looks like '{inferred}', but current model_type is '{self.model_type}'. Rebuilding model to match checkpoint.")
            # pick reasonable defaults per model type
            if inferred == 'SimpleCNN':
                self.model = getattr(model_library, inferred)(output_dim=2, return_sequence=False).to(self.device).float()
            elif inferred == 'ViT':
                self.model = getattr(model_library, inferred)(output_dim=2, return_sequence=True).to(self.device).float()
            else:  # fallback
                self.model = getattr(model_library, inferred)(output_dim=2, return_sequence=True).to(self.device).float()
            self.model.eval()

        # Finally load
        self.model.load_state_dict(state_dict, strict=True)
        self.mylogger(f"[LOADER] Loaded checkpoint from {checkpoint_path} as {self.model.__class__.__name__}")


    def run_model(self, mode='val', return_inputs=False):
        data_loader = self.val_data if mode == 'val' else self.train_data
        results = []

        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                event_seq, mask_seq = batch  # mask_seq shape: (B, T, 1, H, W)
                event_seq = event_seq.to(self.device)
                mask_seq = mask_seq.to(self.device)
                
                if self.model_type == "SimpleCNN":
                    # Already flattened by simplecnn_collate_fn:
                    # event_seq: (B,1,H,W), mask_seq: (B,1,H,W)
                    event_seq = F.interpolate(event_seq, size=(60, 90), mode='bilinear', align_corners=False)
                    X = [event_seq]
                    pred, extras = self.model(X)   # (B,1,H,W)
                elif self.model_type == "ViT":
                    B, T, C, H, W = event_seq.shape
                    event_seq = event_seq.view(B * T, C, H, W)
                    event_in = F.interpolate(event_seq, size=(60, 90), mode='bilinear', align_corners=False)
                    
                    pred, extras = self.model([event_in])   # pred: (B*T,1,60,90)
                else:
                    # Sequential models (e.g. LSTMNetVIT) expect (B,T,1,H,W)
                    B, T, C, H, W = event_seq.shape
                    print(f"[DEBUG] event_seq shape before resize: {event_seq.shape}")
                    event_seq = event_seq.view(B * T, C, H, W)
                    event_seq = F.interpolate(event_seq, size=(60, 90), mode='bilinear', align_corners=False)
                    event_seq = event_seq.view(B, T, C, 60, 90)

                    X = [event_seq]
                    pred, extras = self.model(X)

                # B, T, C, H, W = mask_seq.shape
                # mask_seq = F.interpolate(mask_seq.view(B*T, C, H, W), size=(60, 90), mode='bilinear', align_corners=True)
                # mask_seq = mask_seq.view(B, T, C, 60, 90)
                # # mask_seq = F.interpolate(mask_seq.view(B*T, C, H, W), size=(90, 60), mode='bilinear', align_corners=True)
                # # mask_seq = mask_seq.view(B, T, C, 90, 60)
                # # mask_seq = mask_seq.view(B * T, C, H, W)  # reshape if model expects (B*T, C, H, W)
                
                # # Forward pass (if LSTMNetVIT expects sequence, pass original shape)
                # # pred, extras = self.model(mask_seq.view(B, T, C, H, W))  # (B, T, C, H, W)
                # # print(f"[DEBUG] Running on batch with mask_seq shape: {mask_seq.shape}")

                # pred, extras = self.model([mask_seq])  # already shape (B, T, C, H, W)
                # # pred, extras = self.model(mask_seq)

                if return_inputs:
                    results.append((None, (pred, extras), (event_seq, None, torch.ones_like(mask_seq), (mask_seq, None))))
                else:
                    results.append((None, (pred, extras), None))

        return results




def argparsing(filename=None, cli_args=None):

    if filename is not None:
        default_config_files = [filename]
    else:
        default_config_files = [f'/home/{uname}/catkin_ws/src/lane_marking_demo/dvs_learning/event2seg/training/configs/train_config_event2seg_on_real.txt']

    import configargparse
    parser = configargparse.ArgumentParser(default_config_files=default_config_files)
    # parser = configargparse.ArgumentParser()  # ← NO default_config_files
    parser.add_argument('--config', is_config_file=True, help='Path to config file', default=default_config_files[0])
    parser.add_argument('--basedir', type=str, default=f'/home/{uname}/catkin_ws/src/lane_marking_demo')
    # parser.add_argument('--dataset', type=str, default='dataset110')
    parser.add_argument(
        '--real_data_root',
        type=str,
        required=True,
        help='Root directory containing only the real trajectory folders with masks/'
    )
    # --- in argparsing() ---
    parser.add_argument(
        '--event_npy_path',
        type=str,
        default=f'/home/{uname}/catkin_ws/src/lane_marking_demo/dvs_learning/data/evs_frames.npy',
        help='Path to evs_frames.npy containing object array of per-trajectory event-frame tensors'
    )
    parser.add_argument(
        '--align_mode',
        type=str,
        choices=['drop_first_image', 'drop_last_image'],
        default='drop_first_image',
        help='How to align images with events when there is one more image than event frame'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.25,
        help='Fraction of trajectories to use for validation'
    )
    parser.add_argument('--dataset_range', type=str,
                    help='Comma-separated list of dataset numbers and/or ranges, e.g. "132-135,152-155"')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--model_type', type=str, default='ViT')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        nargs='*',
        default=None,
        help='Optional pretrained checkpoint path. If provided, fine-tune from simulation checkpoint. If omitted, train from scratch baseline.'
    )
    parser.add_argument('--combine_checkpoints', action='store_true', help='Whether to combine multiple checkpoints')
    parser.add_argument("--seq_len", type=int, default=16,
                    help="Sequence length for window-based sampling")
    parser.add_argument("--batch_size", type=int, default=8,
                    help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4,
                    help="Number of workers for data loading")

    return parser.parse_args(cli_args)

if __name__ == '__main__':
    args = argparsing()
    learner = TRAINER(args)
    learner.train()
