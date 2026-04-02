import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import cv2
import csv
import glob, os, time
from os.path import join as opj
import random
import getpass
uname = getpass.getuser()

class DrivingDataset(Dataset):
    def __init__(self, masks, cmds, meta, lengths, mean, std, seq_len=16, transform=None):
        self.masks = masks
        self.cmds = cmds
        self.meta = meta
        self.lengths = lengths
        self.mean = mean
        self.std = std
        self.seq_len = seq_len
        self.transform = transform
        
        # store tensors for TRAINER access
        self.cmd_mean = torch.tensor(mean, dtype=torch.float32)
        self.cmd_std = torch.tensor(std, dtype=torch.float32)

        # # Define paths
        # event_dir = os.path.join(root_dir, "events")
        # rgb_dir = os.path.join(root_dir, "rgb")
        # mask_dir = os.path.join(root_dir, "masks")
        # cmd_path = os.path.join(root_dir, "control_odometry.csv")

        # # Load control_odometry.csv into a dict
        # self.cmd_dict = {}
        # all_cmds = []
        # with open(cmd_path, "r") as f:
        #     reader = csv.DictReader(f)
        #     for row in reader:
        #         # self.cmd_dict[row["timestamp"]] = row
        #         # Store [steering, steer_vel, speed, accel, jerk] as floats
        #         # 5 outputs
        #         # cmd = [
        #         #     float(row["steering_angle"]),
        #         #     float(row["steering_angle_velocity"]),
        #         #     float(row["speed"]),
        #         #     float(row["acceleration"]),
        #         #     float(row["jerk"])
        #         # ]
        #         # cmd = [
        #         #     float(row["steering_angle"]),
        #         #     float(row["steering_angle_velocity"]),
        #         #     float(row["speed"]),
        #         #     float(row["acceleration"]),
        #         #     float(row["jerk"])
        #         # ]

        #         # 2 outputs
        #         cmd = [
        #             float(row["steering_angle"]),
        #             float(row["speed"])
        #         ]
        #         self.cmd_dict[row["timestamp"]] = cmd
        #         all_cmds.append(cmd)

        # # Collect samples based on event image timestamps
        # # image_files = sorted(os.listdir(event_dir))
        # image_files = sorted(os.listdir(mask_dir))
        # self.samples = []
        # for fname in image_files:
        #     timestamp = os.path.splitext(fname)[0]
        #     if timestamp not in self.cmd_dict:
        #         continue
        #     # self.samples.append({
        #     #     "event_path": os.path.join(event_dir, fname),
        #     #     "rgb_path": os.path.join(rgb_dir, fname),
        #     #     "mask_path": os.path.join(mask_dir, fname),
        #     #     "cmd": self.cmd_dict[timestamp],
        #     # })
        #     self.samples.append({
        #         "mask_path": os.path.join(mask_dir, fname),
        #         "cmd": self.cmd_dict[timestamp]
        #     })
        
        # if len(self.samples) == 0:
        #     raise RuntimeError(f"[DrivingDataset] No valid samples found in {root_dir}. Check if 'events/' and 'control_odometry.csv' match.")

        # # --- Compute normalization stats for [steering, steer_vel, speed, accel, jerk] ---
        # # all_cmds = []
        # # for s in self.samples:
        # #     row = s["cmd"]
        # #     all_cmds.append([
        # #         float(row["steering_angle"]),
        # #         float(row["steering_angle_velocity"]),
        # #         float(row["speed"]),
        # #         float(row["acceleration"]),
        # #         float(row["jerk"]),
        # #     ])
        # # all_cmds = np.stack(all_cmds, axis=0)       # shape (N,5)
        # all_cmds = torch.tensor(all_cmds, dtype=torch.float32)  # shape (N,5)
        # means = all_cmds.mean(axis=0)               # shape (5,)
        # stds  = all_cmds.std(axis=0) + 1e-6         # shape (5,) add epsilon to avoid div0

        # # store as tensors on CPU
        # self.cmd_mean = means
        # self.cmd_std  = stds

    def __len__(self):
        # return len(self.samples) - self.seq_len+ 1
        return len(self.cmds) - self.seq_len + 1

    # def __getitem__(self, idx):
    #     s = self.samples[idx]

    #     # Load images
    #     event = cv2.imread(s["event_path"], cv2.IMREAD_GRAYSCALE)
    #     rgb = cv2.imread(s["rgb_path"], cv2.IMREAD_COLOR)
    #     mask = cv2.imread(s["mask_path"], cv2.IMREAD_GRAYSCALE)

    #     # Load control + odometry command vector
    #     cmd_row = s["cmd"]
    #     cmd = torch.tensor([
    #         float(cmd_row["steering_angle"]),
    #         float(cmd_row["steering_angle_velocity"]),
    #         float(cmd_row["speed"]),
    #         float(cmd_row["acceleration"]),
    #         float(cmd_row["jerk"]),
    #         float(cmd_row["position_x"]),
    #         float(cmd_row["position_y"]),
    #         float(cmd_row["position_z"]),
    #         float(cmd_row["orientation_x"]),
    #         float(cmd_row["orientation_y"]),
    #         float(cmd_row["orientation_z"]),
    #         float(cmd_row["orientation_w"]),
    #     ], dtype=torch.float)

    #     # Normalize images
    #     event = event / 255.0
    #     rgb = rgb / 255.0
    #     mask = mask / 255.0

    #     # Convert to tensors
    #     event = torch.from_numpy(event).float().unsqueeze(0)         # 1 x H x W
    #     rgb = torch.from_numpy(rgb).float().permute(2, 0, 1)         # 3 x H x W
    #     mask = torch.from_numpy(mask).float().unsqueeze(0)           # 1 x H x W
    #     desired_vel = cmd[2:3]
    #     quat = cmd[8:12]

    #     return [event, desired_vel, quat]

    def __getitem__(self, idx):
        # # sequence of masks
        # mask_seq = self.masks[idx: idx + self.seq_len]   # (seq_len, H, W)

        # # sequence of commands
        # cmd_seq = self.cmds[idx: idx + self.seq_len]     # (seq_len, cmd_dim)

        # # normalize commands
        # cmd_seq = (cmd_seq - self.mean) / self.std

        # # optional: apply transform to each frame
        # if self.transform:
        #     mask_seq = [self.transform(m) for m in mask_seq]

        # # convert to tensors
        # mask_seq = torch.tensor(mask_seq, dtype=torch.float32) # (H, W)
        # mask_seq = mask_seq.unsqueeze(0) # (1, H, W) -> adds channel
        # cmd_seq = torch.tensor(cmd_seq, dtype=torch.float32)

        # Collect a sequence of frames and commands
        mask_seq = []
        cmd_seq = []
        for t in range(self.seq_len):
            mask = self.masks[idx + t]      # (1, H, W)
            cmd = self.cmds[idx + t]                   # (cmd_dim,)
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)
            if isinstance(cmd, np.ndarray):
                cmd = torch.from_numpy(cmd).float()  # (cmd_dim,)
            # Normalize command
            cmd = (cmd - self.cmd_mean) / self.cmd_std
            
            mask_seq.append(mask)
            cmd_seq.append(cmd)

        mask_seq = torch.stack(mask_seq, dim=0)        # (T, 1, H, W)
        cmd_seq = torch.stack(cmd_seq, dim=0)          # (T, cmd_dim)
        
        return mask_seq, cmd_seq
    
    # def __getitem__(self, idx):
    #     s = self.samples[idx]
    #     center = idx
    #     half = self.seq_len // 2

    #     # get indices around center
    #     start = max(0, center - half)
    #     end = start + self.seq_len
    #     if end > len(self.samples):
    #         end = len(self.samples)
    #         start = max(0, end - self.seq_len)

    #     # … load & normalize mask as before …
    #     mask_seq = []
    #     cmd_seq = []
    #     for i in range(start, end):
    #         mask = cv2.imread(self.samples[i]["mask_path"], cv2.IMREAD_GRAYSCALE) / 255.0
    #         # mask = cv2.resize(mask, (60, 90))
    #         mask = torch.from_numpy(mask).float().unsqueeze(0)  # (1,H,W)
    #         mask_seq.append(mask)

    #         cmd = torch.tensor(self.samples[i]["cmd"], dtype=torch.float)
    #         cmd_norm = (cmd - self.mean) / self.std  # normalize to zero mean, unit var
    #         cmd_seq.append(cmd_norm)
        

    #     # Pad if necessary to get full length
    #     while len(mask_seq) < self.seq_len:
    #         mask_seq.append(mask_seq[-1].clone())
    #         cmd_seq.append(cmd_seq[-1].clone())

    #     mask_seq = torch.stack(mask_seq, dim=0)  # shape: [T, 1, H, W]
    #     cmd_seq = torch.stack(cmd_seq, dim=0)  # shape: [T, 5]
        
    #     return mask_seq.float(), cmd_seq.float()
        # if self.model_type == "SimpleCNN":
        #     return mask_seq.float(), cmd_seq.float()  # each sample returns (T,1,H,W), (T,5)
        # else:
        #     return mask_seq.float(), cmd_seq.float()

        # mask_seq = torch.stack(mask_seq)  # shape: [T, 1, H, W]

        # # Build your 5‑D target: [steering_angle, steering_angle_velocity, speed, acceleration, jerk]
        # # cmd = s["cmd"]
        # # drive = torch.tensor([
        # #     float(cmd["steering_angle"]),
        # #     float(cmd["steering_angle_velocity"]),
        # #     float(cmd["speed"]),
        # #     float(cmd["acceleration"]),
        # #     float(cmd["jerk"]),
        # # ], dtype=torch.float) 

        # cmd = torch.tensor(s["cmd"], dtype=torch.float)
        # print(f"cmd.shape = {cmd.shape}")
        # # normalize to zero mean, unit var
        # drive_norm = (cmd - self.cmd_mean) / self.cmd_std

        # return mask_seq, drive_norm

    # def __getitem__(self, idx):
    #     # Treat idx as the START of the window: [idx : idx+seq_len)
    #     start = idx
    #     end   = idx + self.seq_len  # guaranteed <= len(self.samples)

    #     mask_seq, cmd_seq = [], []
    #     for i in range(start, end):
    #         mask = cv2.imread(self.samples[i]["mask_path"], cv2.IMREAD_GRAYSCALE) / 255.0
    #         mask = torch.from_numpy(mask).float().unsqueeze(0)  # (1,H,W)
    #         mask_seq.append(mask)

    #         cmd = torch.tensor(self.samples[i]["cmd"], dtype=torch.float)
    #         cmd_seq.append((cmd - self.cmd_mean) / self.cmd_std)

    #     mask_seq = torch.stack(mask_seq, dim=0)  # (T,1,H,W)
    #     cmd_seq  = torch.stack(cmd_seq,  dim=0)  # (T,2)  # your current 2-dim setup
    #     return mask_seq.float(), cmd_seq.float()

def collate_fn(batch):
    # if isinstance(batch[0][0], torch.Tensor) and batch[0][0].dim() == 3:
    #     # SimpleCNN: already (1,H,W), (5)
    #     images, cmds = zip(*batch)
    #     return torch.stack(images, 0), torch.stack(cmds, 0)
    # else:
    # Flatten sequence dimension
    all_images, all_cmds = [], []
    for images, cmds in batch:
        all_images.append(images)  # (T,1,H,W)
        all_cmds.append(cmds)      # (T,5)
    return torch.cat(all_images, dim=0), torch.cat(all_cmds, dim=0)


# def dataloader(data_dirs, val_split=0.1, batch_size=8, num_workers=4, collate_fn=None):
#     datasets = [DrivingDataset(d) for d in data_dirs]
#     full_dataset = torch.utils.data.ConcatDataset(datasets)

#     val_size = int(len(full_dataset) * val_split)
#     train_size = len(full_dataset) - val_size

#     train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

#     return train_loader, val_loader
#     # dataset = DrivingDataset(data_dir)
#     # val_size = int(len(dataset) * val_split)
#     # train_size = len(dataset) - val_size

#     # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#     # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     # return train_loader, val_loader

# --- at the top of dataloader(...) keep your imports as-is ---

def dataloader(data_dir, val_split=0.1, short=0, seed=None, train_val_dirs=None):
    cropHeight = 60
    cropWidth = 90

    # ==== NEW: deterministic shuffle banner ====
    if seed is not None:
        random.seed(seed)

    if train_val_dirs is not None:
        train_dirs, val_dirs = train_val_dirs
        traj_folders = val_dirs + train_dirs   # keep order but doesn’t matter anymore
        use_explicit = True

    else:
        if isinstance(data_dir, list):
            traj_folders = data_dir
        else:
            traj_folders = sorted(glob.glob(opj(data_dir, 'dataset*')))
        random.shuffle(traj_folders)  # shuffle the folders
        use_explicit = False
        # # Old frame-level split
        # n_total = traj_meta.shape[0]
        # n_val = max(1, int(val_split * n_total))
        # n_train = n_total - n_val

        # idxs = np.arange(n_total)
        # if seed is not None:
        #     np.random.seed(seed + i)
        # np.random.shuffle(idxs)

        # val_idx = idxs[:n_val]
        # train_idx = idxs[n_val:]

        # train_ims_all.append(traj_ims[train_idx])
        # val_ims_all.append(traj_ims[val_idx])
        # train_meta_all.append(traj_meta[train_idx])
        # val_meta_all.append(traj_meta[val_idx])
        # train_cmds_all.append(traj_meta[train_idx, 2:4])
        # val_cmds_all.append(traj_meta[val_idx, 2:4])
        # train_quats_all.append(traj_meta[train_idx, 3:7])
        # val_quats_all.append(traj_meta[val_idx, 3:7])
        # use_explicit = False

    if short > 0:
        assert short <= len(traj_folders), f"short={short} is greater than the number of folders={len(traj_folders)}"
        traj_folders = traj_folders[:short]

    desired_vels = []
    traj_ims_full = []
    traj_meta_full = []
    curr_quats = []

    # (optional) keep per-folder lengths in the same order as traj_folders
    per_folder_lengths = []

    start_dataloading = time.time()
    skippedImages = 0
    skippedFolders = 0
    collisionImages = 0
    collisionFolders = 0

    train_ims_all, val_ims_all = [], []
    train_meta_all, val_meta_all = [], []
    train_cmds_all, val_cmds_all = [], []
    train_quats_all, val_quats_all = [], []

    for i, traj_folder in enumerate(traj_folders):
        if len(traj_folders)//10 > 0 and i % (len(traj_folders)//10) == 0:
            print(f'[DATALOADER] Loading folder {os.path.basename(traj_folder)}, folder # {i+1}/{len(traj_folders)}, time elapsed {time.time()-start_dataloading:.2f}s')

        im_files = sorted(glob.glob(opj(traj_folder, 'masks', '*.png')))
        if len(im_files) == 0:
            print(f'[DATALOADER] No images in {os.path.basename(traj_folder)}, skipping')
            continue

        csv_file = 'control_odometry.csv'
        traj_meta = np.genfromtxt(opj(traj_folder, csv_file), delimiter=',', dtype=np.float64)[1:]
        traj_meta[:,-1] = np.int32(np.genfromtxt(opj(traj_folder, csv_file), delimiter=',', dtype="bool")[1:,-1])

        if np.isnan(traj_meta).any():
            print(f'[DATALOADER] NaN in {os.path.basename(traj_folder)}, skipping')
            traj_meta = traj_meta[:,:-1]

        traj_ims = np.asarray([cv2.imread(im_file, cv2.IMREAD_GRAYSCALE) for im_file in im_files], dtype=np.float32) / 255.0

        # --- split ---
        if use_explicit:  # trajectory-level split
            if traj_folder in val_dirs:
                val_ims_all.append(traj_ims)
                val_meta_all.append(traj_meta)
                val_cmds_all.append(traj_meta[:, 2:4])
                val_quats_all.append(traj_meta[:, 3:7])
            else:
                train_ims_all.append(traj_ims)
                train_meta_all.append(traj_meta)
                train_cmds_all.append(traj_meta[:, 2:4])
                train_quats_all.append(traj_meta[:, 3:7])
        else:
            n_total = traj_meta.shape[0]
            n_val = max(1, int(val_split * n_total))
            idxs = np.arange(n_total)
            if seed is not None:
                np.random.seed(seed + i)
            np.random.shuffle(idxs)
            val_idx, train_idx = idxs[:n_val], idxs[n_val:]

            train_ims_all.append(traj_ims[train_idx])
            val_ims_all.append(traj_ims[val_idx])
            train_meta_all.append(traj_meta[train_idx])
            val_meta_all.append(traj_meta[val_idx])
            train_cmds_all.append(traj_meta[train_idx, 2:4])
            val_cmds_all.append(traj_meta[val_idx, 2:4])
            train_quats_all.append(traj_meta[train_idx, 3:7])
            val_quats_all.append(traj_meta[val_idx, 3:7])

        if traj_ims.shape[0] != traj_meta.shape[0]:
            last_im_timestamp = os.path.basename(im_files[-1])[:-4]
            if float(last_im_timestamp) > traj_meta[-1, 1]:
                traj_ims = traj_ims[:-1]
                print(f'[DATALOADER] Extra image found at end of data, cutting it from {os.path.basename(traj_folder)}')
            if traj_ims.shape[0] != traj_meta.shape[0]:
                print(f'[DATALOADER] Number of images and telemetry still do not match in {os.path.basename(traj_folder)}, skipping')
                skippedFolders += 1
                skippedImages += int(len(traj_meta[:,0]))
                continue
        
        # for ii in range(traj_meta.shape[0]):
        #     desired_vels.append(traj_meta[ii, 2:4])
        #     q = traj_meta[ii, 3:7]
        #     curr_quats.append(q)

        # try:
        #     traj_ims_full.append(traj_ims)
        #     traj_meta_full.append(traj_meta)
        #     per_folder_lengths.append(traj_meta.shape[0])
        # except:
        #     print(f'[DATALOADER] {traj_ims.shape}')
        #     print(f"[DATALOADER] Suspected empty image, folder {os.path.basename(traj_folder)}")

    print("[DATALOADER] Data loading complete, processing splits...")
    traj_ims_train = np.concatenate(train_ims_all)
    traj_ims_val   = np.concatenate(val_ims_all)
    print("[DATALOADER] Finished concatenating images...")

    traj_meta_train = np.concatenate(train_meta_all)
    traj_meta_val   = np.concatenate(val_meta_all)
    print("[DATALOADER] Finished concatenating meta...")

    desired_vels_train = np.concatenate(train_cmds_all)
    desired_vels_val   = np.concatenate(val_cmds_all)
    print("[DATALOADER] Finished concatenating cmds...")

    curr_quats_train = np.concatenate(train_quats_all)
    curr_quats_val   = np.concatenate(val_quats_all)

    print("[DATALOADER] Finished splitting data into train/val sets.")
    print(skippedFolders, skippedImages)
    print(collisionFolders, collisionImages)

    print("[DATALOADER] Finished printing stats, now normalizing ctbr...")

    # print("[ANALYZER] Analyzing the data....")
    # traj_lengths = np.array(per_folder_lengths, dtype=np.int32)

    # traj_ims_full = np.concatenate(traj_ims_full).reshape(-1, cropHeight, cropWidth)
    # traj_meta_full = np.concatenate(traj_meta_full).reshape(-1, traj_meta.shape[-1])
    # desired_vels = np.array(desired_vels)
    # curr_quats = np.array(curr_quats)

    # --- ctbr normalization (already on *_full) ---
    # stats_ctbr = np.zeros((4, 2))
    # stats_ctbr[0, :] = np.mean(traj_meta_full[:, 16]), np.std(traj_meta_full[:, 16])
    # stats_ctbr[1, :] = np.mean(traj_meta_full[:, 17]), np.std(traj_meta_full[:, 17])
    # stats_ctbr[2, :] = np.mean(traj_meta_full[:, 18]), np.std(traj_meta_full[:, 18])
    # stats_ctbr[3, :] = np.mean(traj_meta_full[:, 19]), np.std(traj_meta_full[:, 19])

    # traj_meta_full[:, 16] = (traj_meta_full[:, 16] - stats_ctbr[0, 0]) / (2 * stats_ctbr[0, 1] + 1e-12)
    # traj_meta_full[:, 17] = (traj_meta_full[:, 17] - stats_ctbr[1, 0]) / (2 * stats_ctbr[1, 1] + 1e-12)
    # traj_meta_full[:, 18] = (traj_meta_full[:, 18] - stats_ctbr[2, 0]) / (2 * stats_ctbr[2, 1] + 1e-12)
    # traj_meta_full[:, 19] = (traj_meta_full[:, 19] - stats_ctbr[3, 0]) / (2 * stats_ctbr[3, 1] + 1e-12)

    # curr_ctbr = traj_meta_full[:, 16:20]

    # ==== REMOVED: the second redundant stats_ctbr block on 'traj_meta' (stale variable) ====

    # --- by-folder split: first chunk == VAL, rest == TRAIN ---
    # num_val_trajs = int(val_split * len(traj_lengths))
    # print(f"[SPLIT] Using {num_val_trajs} trajectories for validation, {len(traj_lengths) - num_val_trajs} for training.")
    # val_idx = np.sum(traj_lengths[:num_val_trajs], dtype=np.int32)

    # traj_meta_val   = traj_meta_full[:val_idx]
    # traj_meta_train = traj_meta_full[val_idx:]
    # traj_ims_val    = traj_ims_full[:val_idx]
    # traj_ims_train  = traj_ims_full[val_idx:]
    # traj_lengths_val   = traj_lengths[:num_val_trajs]
    # traj_lengths_train = traj_lengths[num_val_trajs:]
    # desired_vels_val   = desired_vels[:val_idx]
    # desired_vels_train = desired_vels[val_idx:]
    # curr_quats_val     = curr_quats[:val_idx]
    # curr_quats_train   = curr_quats[val_idx:]
    # # curr_ctbr_val      = curr_ctbr[:val_idx]
    # # curr_ctbr_train    = curr_ctbr[val_idx:]

    # # --- PRINT the actual split so you can eyeball it ---
    # val_folders   = traj_folders[:num_val_trajs]
    # train_folders = traj_folders[num_val_trajs:]

    # print("\n[SPLIT][SUMMARY]")
    # print(f"  VAL trajectories   ({len(val_folders)}):")
    # for p, n in zip(val_folders, traj_lengths_val):
    #     print(f"    - {os.path.basename(p):<20}  frames={int(n)}")
    # print(f"  TRAIN trajectories ({len(train_folders)}):")
    # for p, n in zip(train_folders, traj_lengths_train):
    #     print(f"    - {os.path.basename(p):<20}  frames={int(n)}")
    # print(f"  Totals: VAL frames={int(traj_lengths_val.sum())}, TRAIN frames={int(traj_lengths_train.sum())}\n")

    # compute lengths per trajectory
    traj_lengths_train = np.array([len(x) for x in train_ims_all], dtype=np.int32)
    traj_lengths_val   = np.array([len(x) for x in val_ims_all], dtype=np.int32)

    # train_cmds = np.concatenate(desired_vels_train, axis=0)  # shape (N, cmd_dim)
    train_cmds = desired_vels_train
    cmd_mean = np.mean(train_cmds, axis=0)
    cmd_std  = np.std(train_cmds, axis=0) + 1e-8  # avoid div by zero

    # Return (train, val, is_png_flag, (train_folder_list, val_folder_list))
    # return (traj_meta_train, traj_ims_train, traj_lengths_train, desired_vels_train, curr_quats_train), \
    #        (traj_meta_val,   traj_ims_val,   traj_lengths_val,   desired_vels_val,   curr_quats_val), \
    #        1, \
    #        (train_folders, val_folders), \
    #        (cmd_mean, cmd_std)

    print(f"[DATALOADER] Finished loading data, train frames: {len(traj_ims_train)}, val frames: {len(traj_ims_val)}")

    return (traj_meta_train, traj_ims_train, traj_lengths_train, desired_vels_train, curr_quats_train), \
           (traj_meta_val,   traj_ims_val,   traj_lengths_val,   desired_vels_val,   curr_quats_val), \
           1, \
           None, \
           (cmd_mean, cmd_std)