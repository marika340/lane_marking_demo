import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join as opj
import time
import cv2

def eval_masks_and_cmds(learner, checkpoint_path, dataSetstoTest=5, load_ckpt=True):
    if load_ckpt:
        learner.load_from_checkpoint(checkpoint_path)

    learner.model.eval()
    learner.mylogger(f"[EVAL] Running eval_masks_and_cmds on {checkpoint_path}...")

    with torch.no_grad():
        # num_to_plot = min(dataSetstoTest, len(learner.val_dirs))  # <- use val_dirs to count available trajectories
        num_to_plot = dataSetstoTest

        for i in range(num_to_plot):
            # Run model on the i-th validation trajectory
            # loss, (pred, extras), (traj_input_ims, traj_input_evs, desvel, gt) = learner.run_model(
            #     i, mode='val', return_inputs=True, seq_input=False, do_step=False
            # )

            # loss, (pred, extras), (traj_input_ims, traj_input_evs, desvel, gt) = learner.run_model(
            #     mode='val', return_inputs=True
            # )

            results = learner.run_model(mode='val', return_inputs=True)

            for i, (loss, (pred, extras), (mask, _, desvel, (gt, _))) in enumerate(results):
                print(f"--- Trajectory {i} ---")
                for step in range(pred.shape[0]):
                    pred_vel = pred[step][:3].detach().cpu().numpy()
                    print(f"Step {step:03d}: Predicted vel = [{pred_vel[0]:.3f}, {pred_vel[1]:.3f}, {pred_vel[2]:.3f}]")

                # Save input mask image
                input_mask = mask[0].detach().cpu().numpy() * 255.0  # (1,H,W) → (H,W)
                input_mask = input_mask.squeeze().astype(np.uint8)
                save_path = os.path.join(learner.workspace, f"traj_{i}_input_mask.png")
                cv2.imwrite(save_path, input_mask)
                print(f"[EVAL] Saved mask image to {save_path}")

            pred_vel = pred
            pred_vel = pred_vel * desvel  # scale to match ground truth magnitude

            # Print predicted commands
            print(f"\n--- Trajectory {i} ---")
            for t in range(pred_vel.shape[0]):
                vel = pred_vel[t].cpu().numpy()
                print(f"Step {t:03d}: Predicted vel = [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]")

            # Save first non-blank input mask image
            traj_input_np = traj_input_ims.cpu().numpy()
            mean_per_frame = traj_input_np.mean(axis=(2, 3))
            first_idx = np.where(mean_per_frame < 0.95)[0][0] if (mean_per_frame < 0.95).any() else 0
            img = traj_input_np[first_idx, 0]

            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.title(f"Trajectory {i} - Input Mask")
            save_path = opj(learner.workspace, f"traj_{i}_input_mask.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            learner.mylogger(f"[EVAL] Saved mask image to {save_path}")
