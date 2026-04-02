import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import os, glob, random, time
import numpy as np
import cv2
from os.path import join as opj

# ============================================================
# SegDataset: lazy loading from file paths + metadata
# ============================================================
class SegDataset(Dataset):
    def __init__(self, folders, seq_len=16, transform=None):
        """
        folders: list of dataset folders (each with masks/ and control_odometry.csv)
        seq_len: sequence length
        """
        self.samples = []
        self.seq_len = seq_len
        self.transform = transform

        for folder in folders:
            event_dir = opj(folder, "events")
            mask_dir = opj(folder, "masks")
            if not os.path.exists(event_dir) or not os.path.exists(mask_dir):
                print(f"[DrivingDataset] Skipping {folder}, missing files")
                continue

            # load metadata
            event_files = sorted(glob.glob(opj(event_dir, "*.png")))
            mask_files = sorted(glob.glob(opj(mask_dir, "*.png")))

            # # Filter out rows where the third column (steering_angle) is outside [-10, 10]
            # if len(traj_meta.shape) < 2 or len(mask_files) == 0:
            #     continue
            # valid_mask = (traj_meta[:,2] >= -10.0) & (traj_meta[:,2] <= 10.0)
            # traj_meta = traj_meta[valid_mask]
            # Also filter mask_files accordingly
            # mask_files = [f for i, f in enumerate(mask_files) if valid_mask[i]]
            n = min(len(event_files), len(mask_files))

            # build samples (start indices for sequences)
            for i in range(0, n - seq_len + 1, seq_len):
                self.samples.append((event_files[i:i+seq_len], mask_files[i:i+seq_len])) 

        print(f"[DrivingDataset] Built {len(self.samples)} samples from {len(folders)} folders")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        event_files, mask_files = self.samples[idx]

        event_seq = []
        for f in event_files:
            event = cv2.imread(f) # BGR order
            if event is None:
                raise RuntimeError(f"Failed to load {f}")
            # Set binary (0, 1). 1 where B or R channel is 255
            event = ((event[:,:,0] == 255) | (event[:,:,2] == 255)).astype(np.float32)
            event = torch.from_numpy(event).unsqueeze(0)  # (1,H,W)
            if self.transform:
                event = self.transform(event)
            event_seq.append(event)
        
        event_seq = torch.stack(event_seq, dim=0)  # (T,1,H,W)

        mask_seq = []
        for f in mask_files:
            mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Failed to load {f}")
            mask = mask.astype(np.float32) / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)  # (1,H,W)
            if self.transform:
                mask = self.transform(mask)
            mask_seq.append(mask)

        mask_seq = torch.stack(mask_seq, dim=0)  # (T,1,H,W)

        return event_seq, mask_seq


# ============================================================
# Collate function: stack sequences into batch
# ============================================================
def collate_fn(batch):
    all_events, all_masks = zip(*batch)  # each is (T,1,H,W), (T,1,H,W)
    return torch.stack(all_events, dim=0), torch.stack(all_masks, dim=0)


# ============================================================
# dataloader function
# ============================================================
def dataloader(data_dir, val_split=0.1, seq_len=16, batch_size=8, num_workers=4, seed=None, train_val_dirs=None):
    """
    Builds train/val DataLoaders from dataset folders
    """
    if seed is not None:
        random.seed(seed)
    # random.shuffle(traj_folders)

    if train_val_dirs is not None:
        train_folders, val_folders = train_val_dirs
        traj_folders = val_folders + train_folders   # keep order but doesn’t matter anymore
        use_explicit = True

    else:
        if isinstance(data_dir, list):
            traj_folders = data_dir
        else:
            traj_folders = sorted(glob.glob(opj(data_dir, 'dataset*')))
        random.shuffle(traj_folders)  # shuffle the folders
        use_explicit = False

        n_val = max(1, int(len(traj_folders) * val_split))
        val_folders = traj_folders[:n_val]
        train_folders = traj_folders[n_val:]

    # --- compute normalization stats from train set only ---
    # print("[DATALOADER] Scanning commands for normalization stats...")
    # all_cmds = []
    # for folder in train_folders:
    #     csv_file = opj(folder, "control_odometry.csv")
    #     if os.path.exists(csv_file):
    #         traj_meta = np.genfromtxt(csv_file, delimiter=',', dtype=np.float64)[1:]
    #         if len(traj_meta.shape) < 2:
    #             print(f"[DATALOADER] Skipping {folder}, no data found")
    #             continue
    #         if traj_meta.shape[0] > 0:
    #             all_cmds.append(traj_meta[:, 2:4])  # steering_angle, speed
    # all_cmds = np.concatenate(all_cmds, axis=0) if len(all_cmds) > 0 else np.zeros((1,2))
    # cmd_mean = all_cmds.mean(axis=0)
    # cmd_std  = all_cmds.std(axis=0) + 1e-8
    # print(f"[DATALOADER] cmd_mean={cmd_mean}, cmd_std={cmd_std}")

    # --- datasets ---
    train_dataset = SegDataset(train_folders, seq_len=seq_len)
    val_dataset   = SegDataset(val_folders,   seq_len=seq_len)

    # --- loaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    return train_loader, val_loader
