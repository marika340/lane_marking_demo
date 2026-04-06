import os
import glob
import random
from os.path import join as opj

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SegDataset(Dataset):
    def __init__(self, folders, evs_subset, seq_len=16,
                 align_mode='drop_first_image', transform=None):
        self.seq_len = seq_len
        self.transform = transform
        self.align_mode = align_mode
        self.samples = []
        self.evs_all = evs_subset

        if len(self.evs_all) != len(folders):
            raise ValueError(
                f"Mismatch between number of folders ({len(folders)}) and "
                f"number of trajectories provided ({len(self.evs_all)})"
            )

        for traj_idx, folder in enumerate(folders):
            mask_dir = opj(folder, "masks")
            if not os.path.exists(mask_dir):
                raise ValueError(f"Missing masks directory: {mask_dir}")

            mask_files = sorted(glob.glob(opj(mask_dir, "*.png")))
            ev_frames = self.evs_all[traj_idx]

            if ev_frames is None:
                raise ValueError(f"Trajectory {traj_idx} in npy is None")

            n_events = len(ev_frames)
            n_masks = len(mask_files)

            if n_masks == n_events + 1:
                if align_mode == 'drop_first_image':
                    aligned_masks = mask_files[1:]
                elif align_mode == 'drop_last_image':
                    aligned_masks = mask_files[:-1]
                else:
                    raise ValueError(f"Unknown align_mode: {align_mode}")
            elif n_masks == n_events:
                aligned_masks = mask_files
                print(f"[SegDataset] Warning: {folder} already has equal counts")
            else:
                raise ValueError(
                    f"{folder}: got {n_masks} masks and {n_events} event frames; "
                    f"expected #masks == #events or #events + 1"
                )

            if len(aligned_masks) != n_events:
                raise ValueError(
                    f"{folder}: alignment failed, got {len(aligned_masks)} masks "
                    f"for {n_events} events"
                )

            # non-overlapping windows, same as your original code
            for start in range(0, n_events - seq_len + 1, seq_len):
                self.samples.append({
                    "traj_idx": traj_idx,
                    "start": start,
                    "mask_files": aligned_masks[start:start + seq_len]
                })

        print(f"[SegDataset] Built {len(self.samples)} samples from {len(folders)} folders")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        traj_idx = sample["traj_idx"]
        start = sample["start"]
        mask_files = sample["mask_files"]

        ev_frames = self.evs_all[traj_idx][start:start + self.seq_len]

        event_seq = []
        for ev in ev_frames:
            ev = np.asarray(ev, dtype=np.float32)
            if ev.ndim != 2:
                raise RuntimeError(f"Expected event frame shape (H,W), got {ev.shape}")
            event = torch.from_numpy(ev).unsqueeze(0)  # (1,H,W)
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


def collate_fn(batch):
    all_events, all_masks = zip(*batch)
    return torch.stack(all_events, dim=0), torch.stack(all_masks, dim=0)


def dataloader(
    data_dir,
    val_split=0.25,
    seq_len=16,
    batch_size=8,
    num_workers=4,
    seed=None,
    event_npy_path=None,
    align_mode='drop_first_image'
):
    """
    data_dir: root containing only the real trajectory folders
              each trajectory folder must contain masks/
    event_npy_path: path to evs_frames.npy
    """

    if event_npy_path is None:
        raise ValueError("event_npy_path must be provided")

    if not os.path.exists(data_dir):
        raise ValueError(f"data_dir does not exist: {data_dir}")

    if seed is not None:
        random.seed(seed)

    evs_all = np.load(event_npy_path, allow_pickle=True)

    # only keep subfolders that actually contain masks/
    all_folders = sorted([
        f for f in glob.glob(opj(data_dir, "*"))
        if os.path.isdir(f) and os.path.exists(opj(f, "masks"))
    ])

    print(f"[dataloader] discovered {len(all_folders)} trajectory folders under {data_dir}")
    for f in all_folders:
        print("   ", f)

    if len(all_folders) != len(evs_all):
        raise ValueError(
            f"evs_frames.npy contains {len(evs_all)} trajectories, but "
            f"found {len(all_folders)} trajectory folders under {data_dir}. "
            f"Those counts must match exactly."
        )

    n_total = len(all_folders)
    n_val = max(1, int(round(n_total * val_split)))
    n_train = n_total - n_val

    if n_train <= 0:
        raise ValueError(
            f"val_split={val_split} leaves no training trajectories. "
            f"Need at least 1 train trajectory."
        )

    # preserve order so folder order matches npy order
    train_folders = all_folders[:n_train]
    val_folders = all_folders[n_train:]

    train_evs = evs_all[:n_train]
    val_evs = evs_all[n_train:]

    print("[dataloader] train_folders:")
    for f in train_folders:
        print("   ", f)

    print("[dataloader] val_folders:")
    for f in val_folders:
        print("   ", f)

    train_dataset = SegDataset(
        train_folders,
        evs_subset=train_evs,
        seq_len=seq_len,
        align_mode=align_mode
    )

    val_dataset = SegDataset(
        val_folders,
        evs_subset=val_evs,
        seq_len=seq_len,
        align_mode=align_mode
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader