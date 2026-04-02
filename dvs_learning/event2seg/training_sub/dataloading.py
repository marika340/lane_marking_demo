#!/usr/bin/env python3
"""
dataloading.py: PyTorch Dataset for event→segmentation (event frames to mask)
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class EventSegDataset(Dataset):
    """
    Dataset that returns (event_frame_tensor, mask_tensor) pairs.
    Supports .npy event files and .png masks.
    """
    # def __init__(self, events_dir, mask_dir, transforms=None):
    #     super().__init__()
    #     self.events_dir = events_dir
    #     self.mask_dir = mask_dir
    #     self.transforms = transforms

    #     # gather event files
    #     exts = {'.npy', '.png'}
    #     all_files = sorted(os.listdir(events_dir))
    #     self.event_files = [f for f in all_files if os.path.splitext(f)[1].lower() in exts]
    #     if not self.event_files:
    #         raise RuntimeError(f"No event files found in {events_dir}")

    #     # verify corresponding masks exist
    #     self.mask_files = []
    #     for ev_fn in self.event_files:
    #         base, _ = os.path.splitext(ev_fn)
    #         mask_png = base + '.png'
    #         mask_path = os.path.join(mask_dir, mask_png)
    #         if not os.path.exists(mask_path):
    #             raise FileNotFoundError(f"Mask file {mask_png} not found in {mask_dir}")
    #         self.mask_files.append(mask_png)

    def __init__(self, events_dir, mask_dir):
        # Accept list or single path
        if isinstance(events_dir, str):
            events_dir = [events_dir]
        if isinstance(mask_dir, str):
            mask_dir = [mask_dir]

        self.event_paths = []
        self.mask_paths = []

        for ed, md in zip(events_dir, mask_dir):
            ev_files = sorted(os.listdir(ed))
            mask_files = sorted(os.listdir(md))
            assert len(ev_files) == len(mask_files), f"Mismatched count in {ed} and {md}"
            self.event_paths += [os.path.join(ed, f) for f in ev_files]
            self.mask_paths += [os.path.join(md, f) for f in mask_files]

        assert len(self.event_paths) == len(self.mask_paths)

    def __len__(self):
        return len(self.event_files)

    def __getitem__(self, idx):
        # load event frame
        ev_fn = self.event_files[idx]
        ev_path = os.path.join(self.events_dir, ev_fn)
        ext = os.path.splitext(ev_fn)[1].lower()
        if ext == '.npy':
            ev = np.load(ev_path).astype(np.float32)
        else:
            ev = np.array(Image.open(ev_path).convert('F'), dtype=np.float32)
        ev = torch.from_numpy(ev).unsqueeze(0)  # [1, H, W]

        # load mask
        mask_fn = self.mask_files[idx]
        mask_path = os.path.join(self.mask_dir, mask_fn)
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]

        # resize mask to (60, 90)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(60, 90), mode='bilinear', align_corners=False).squeeze(0)
        # mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(90, 60), mode='bilinear', align_corners=False).squeeze(0)


        # optional transforms
        if self.transforms is not None:
            ev, mask = self.transforms(ev, mask)
        
        return [ev], mask
