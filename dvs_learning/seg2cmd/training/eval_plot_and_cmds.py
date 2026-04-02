import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join as opj
import time
import cv2

# def eval_masks_and_cmds(learner, checkpoint_path, dataSetstoTest=5, load_ckpt=True):
#     if load_ckpt:
#         learner.load_from_checkpoint(checkpoint_path)

#     learner.model.eval()
#     learner.mylogger(f"[EVAL] Running eval_masks_and_cmds on {checkpoint_path}...")

#     with torch.no_grad():
#         results = learner.run_model(mode='val', return_inputs=True)
#         results = results[:dataSetstoTest]

#         for i, (loss, (pred, extras), (mask, _, desvel, (gt, _))) in enumerate(results):
#             print(f"\n--- Trajectory {i} ---")
#             print(f"Pred shape: {pred.shape}")
#             # input_masks = mask.detach().cpu().numpy() * 255.0  # shape: (T, 1, H, W)

#             for step in range(pred.shape[0]):
#                 pred_vel = pred[step][:5].detach().cpu().numpy()
#                 print(f"Step {step:03d}: Predicted vel = [{pred_vel[0]:.3f}, {pred_vel[1]:.3f}, {pred_vel[2]:.3f}, {pred_vel[3]:.3f}, {pred_vel[4]:.3f}]")

#             # Save input mask image
#             input_mask = mask[0].detach().cpu().numpy() * 255.0  # (1,H,W) → (H,W)
#             input_mask = input_mask.squeeze().astype(np.uint8)
#             save_path = os.path.join(learner.workspace, f"traj_{i}_input_mask.png")
#             # Ensure the mask is a 2D numpy array (H, W) or 3D (H, W, 1)
#             if isinstance(input_mask, torch.Tensor):
#                 input_mask = input_mask.squeeze().cpu().numpy()  # remove batch/channel dims
#             elif len(input_mask.shape) == 3 and input_mask.shape[0] == 1:
#                 input_mask = input_mask[0]  # (1, H, W) → (H, W)

#             # Optional: Convert boolean or float to uint8 for saving
#             if input_mask.ndim == 3 and input_mask.shape[0] == 16:
#                 input_mask = input_mask[0]  # Take first timestep: (90, 60)

#             if input_mask.dtype != np.uint8:
#                 input_mask = (input_mask * 255).astype(np.uint8)
#             print(f"input_mask shape: {input_mask.shape}")
#             cv2.imwrite(save_path, input_mask)
#             print(f"[EVAL] Saved mask image to {save_path}")

def eval_masks_and_cmds(learner, checkpoint_path, dataSetstoTest=5, load_ckpt=True):
    if load_ckpt:
        learner.load_from_checkpoint(checkpoint_path)

    learner.model.eval()
    learner.mylogger(f"[EVAL] Running eval_masks_and_cmds on {checkpoint_path}...")

    with torch.no_grad():
        results = learner.run_model(mode='val', return_inputs=True)
        results = results[:dataSetstoTest]

        for traj_idx, (loss, (pred_full, extras), (mask_seq, _, desvel, (gt, _))) in enumerate(results):
            print(f"\n--- Trajectory Part {traj_idx} ---")

            # mask_seq: (1, T, 1, H, W)
            T = mask_seq.shape[1]
            ms = mask_seq.float()
            print(f"traj part {traj_idx} mask checksum:", float(ms.sum()), "mean:", float(ms.mean()))
            # fp = ms[0, 0, 0, :5, :5].detach().cpu().numpy()
            if learner.model_type == "SimpleCNN":
                # ms is (B, C, H, W)
                fp = ms[0, 0, :5, :5].detach().cpu().numpy()
            else:
                # ms is (B, T, C, H, W)
                fp = ms[0, 0, 0, :5, :5].detach().cpu().numpy()
            print("traj part", traj_idx, "first-frame top-left 5x5 fingerprint:\n", np.array2string(fp, precision=3))

            h = None  # hidden state for LSTM
            for step in range(T):
                mask_t = mask_seq[:, step]  # (1, 1, H, W)
                mask_t = mask_t.unsqueeze(1) # (B, 1, C, H, W) — T=1
                mask_t = mask_t.unsqueeze(0) # (1, B, 1, C, H, W)
                # print(f"mask_t.shape = {mask_t.shape}")

                # Forward pass: if model takes h, pass it in
                # out = learner.model(mask_t.to(learner.device).float(), h) if h is not None else learner.model(mask_t.to(learner.device).float())
                if learner.model_type == "SimpleCNN":
                    # Flatten sequence: take each frame as an independent sample
                    B, T, C, H, W = mask_t.shape
                    mask_t = mask_t.view(B*T, C, H, W)   # (B*T,C,H,W)
                    out = learner.model(mask_t.to(learner.device).float())
                else:
                    # Sequential models (e.g. LSTMNetVIT)
                    out = learner.model(mask_t.to(learner.device).float(), h) if h is not None else learner.model(mask_t.to(learner.device).float())

                # Handle possible (pred, h) output
                if isinstance(out, tuple) and len(out) == 2:
                    pred_step, h = out
                else:
                    pred_step = out  # non-LSTM model fallback
                    h = None

                # Extract first 5 dims, unnormalize
                # pred_vel_real = pred_step[0, :5] * learner.cmd_std[:5] + learner.cmd_mean[:5]
                pred_vel_real = pred_step[0, :2] * learner.cmd_std[:2] + learner.cmd_mean[:2]
                vals = pred_vel_real.detach().cpu().numpy()
                # print(f"Step {step:03d}: Predicted vel: [steering angle, steering angle velocity, speed, acceleration, jerk] = [{vals[0]:.3f}, {vals[1]:.3f}, {vals[2]:.3f}, {vals[3]:.3f}, {vals[4]:.3f}]")
                print(f"Step {step:03d}: Predicted vel: [steering angle, speed] = [{vals[0]:.3f}, {vals[1]:.3f}]")
                # gt = desvel[0].cpu().numpy() * learner.cmd_std.cpu().numpy() + learner.cmd_mean.cpu().numpy()
                # pred_denorm = pred[0].detach().cpu().numpy() * learner.cmd_std.cpu().numpy() + learner.cmd_mean.cpu().numpy()
                gt_step = gt[step].cpu().numpy() * learner.cmd_std.cpu().numpy() + learner.cmd_mean.cpu().numpy()
                print("  Ground truth cmd (denorm):", gt_step)
                print("  Predicted cmd (denorm):   ", vals)


            # Save last mask in trajectory for quick viewing
            # last_mask = mask_seq[0, -1, 0].detach().cpu().numpy()
            if learner.model_type == "SimpleCNN":
                # masks are (B,1,H,W)
                last_mask = mask_seq[-1,0].detach().cpu()   # (H,W)
            else:
                # masks are (B,T,1,H,W)
                last_mask = mask_seq[0,-1,0].detach().cpu() # (H,W)

            last_mask = (last_mask.cpu().numpy() * 255).astype(np.uint8)
            save_path = os.path.join(learner.workspace, f"traj_{traj_idx}_last_mask.png")
            cv2.imwrite(save_path, last_mask)
            print(f"[EVAL] Saved last mask image to {save_path}")

        # for i, (loss, (pred, extras), (mask, _, desvel, (gt, _))) in enumerate(results):
        #     print(f"\n--- Trajectory {i} ---")
        #     # print(f"Pred shape: {pred.shape}") # (B, T, D)
        #     input_masks = mask.squeeze(2).detach().cpu().numpy() * 255.0  # shape: (1, T, H, W)
            
        #     # print(f"input_masks shape: {input_masks.shape}")
        #     # mask: (B, T, 1, H, W) tensor on CPU or GPU
        #     ms = mask.float()
        #     print(f"traj {i} mask checksum:", float(ms.sum()), "mean:", float(ms.mean()))
        #     # also print a tiny fingerprint of the first frame
        #     fp = ms[0,0,0,:5,:5].detach().cpu().numpy()
        #     print("traj", i, "first-frame top-left 5x5 fingerprint:\n", np.array2string(fp, precision=3))


        #     # Single-step inference (no loop)
        #     pred_step = pred[0, :5]  # Shape: (5,)
        #     pred_vel_real = pred_step * learner.cmd_std[:5] + learner.cmd_mean[:5]  # tensor math
        #     vals = pred_vel_real.detach().cpu().numpy()
        #     print(f"Predicted vel = "
        #         f"[{vals[0]:.3f}, {vals[1]:.3f}, {vals[2]:.3f}, {vals[3]:.3f}, {vals[4]:.3f}]")

        #     # Save the input mask for this step
        #     # Last timestep mask from first batch element
        #     input_mask = input_masks[0, -1]  # Shape: (60, 90)

        #     # Convert to uint8 for saving
        #     if input_mask.dtype != np.uint8:
        #         input_mask = (input_mask * 255).astype(np.uint8)

        #     save_path = os.path.join(learner.workspace, f"traj_{i}_mask.png")
        #     cv2.imwrite(save_path, input_mask)

        #     # for step in range(pred.shape[1]): #T
        #     #     pred_step = pred[0, step, :5]  # tensor
        #     #     pred_vel_real = pred_step * learner.cmd_std[:5] + learner.cmd_mean[:5]  # tensor math
        #     #     vals = pred_vel_real.detach().cpu().numpy()
        #     #     print(f"Step {step:03d}: Predicted vel = "
        #     #         f"[{vals[0]:.3f}, {vals[1]:.3f}, {vals[2]:.3f}, {vals[3]:.3f}, {vals[4]:.3f}]")
                
        #     #     input_mask = input_masks[0, step]  # shape: (1, H, W)
        #     #     # print(f"input_mask shape: {input_mask.shape}")
        #     #     input_mask = input_mask.squeeze().astype(np.uint8)

        #     #     if input_mask.ndim == 3 and input_mask.shape[0] == 1:
        #     #         input_mask = input_mask[0]  # (1, H, W) → (H, W)

        #     #     if input_mask.dtype != np.uint8:
        #     #         input_mask = (input_mask * 255).astype(np.uint8)

        #     #     save_path = os.path.join(learner.workspace, f"traj_{i}_step_{step:03d}_mask.png")
        #     #     cv2.imwrite(save_path, input_mask)
        #     #     # print(f"[EVAL] Saved mask image to {save_path}")
