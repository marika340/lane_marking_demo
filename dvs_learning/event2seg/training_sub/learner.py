#!/usr/bin/env python3
"""
learner.py: Train an event→segmentation model (ConvUNet, OrigUNet, ViT or LSTMNetVIT) for the event2seg stage.
"""
import os
import configargparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataloading import *
from learner_models import *
from vitfly_models import ViT, LSTMNetVIT


def parse_args():
    parser = configargparse.ArgumentParser(
        description='Event→Segmentation Trainer',
        default_config_files=[os.path.join('configs', 'event2seg_config.txt')]
    )
    parser.add('--config', is_config_file=True, help='Path to config file')

    # Data paths
    # parser.add_argument('--events_dir', type=str, required=True,
    #                     help='Directory containing event .npy or .png files')
    # parser.add_argument('--mask_dir', type=str, required=True,
    #                     help='Directory containing ground-truth mask .png files')
    parser.add_argument('--events_dir', nargs='+', type=str, required=True,
                    help='List of directories with event .npy or .png files')
    parser.add_argument('--mask_dir', nargs='+', type=str, required=True,
                    help='List of directories with ground-truth mask .png files')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')

    # Logging & checkpoints
    parser.add_argument('--logdir', type=str, default='runs/event2seg',
                        help='TensorBoard log directory')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save a checkpoint every N epochs')

    # Model choice
    parser.add_argument('--model_type', type=str, default='ConvUNet',
                        choices=['ConvUNet', 'OrigUNet', 'ViT', 'LSTMNetVIT'],
                        help='Architecture to use for segmentation')

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Prepare output dirs
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.logdir)

    # Dataset & DataLoader
    full_dataset = EventSegDataset(
        events_dir=args.events_dir,
        mask_dir=args.mask_dir
    )
    val_split = 0.1
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    # Instantiate model by original class names
    if args.model_type == "ViT":
        model = ViT(in_channels=1, out_channels=1).to(device)
    elif args.model_type == "LSTMNetVIT":
        model = LSTMNetVIT().to(device)
    elif args.model_type == "ConvUNet":
        model = ConvUNet(num_in_channels=1, num_out_channels=1).to(device)
    elif args.model_type == "OrigUNet":
        model = OrigUNet(num_in_channels=1, num_out_channels=1).to(device)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # Loss & optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        hidden_state = None

        for X, masks in train_loader:
            # events: Tensor[B, H, W] or [B, 1, H, W]; ensure [B, 1, H, W]
            # ensure events is [B,1,H,W]
            X = [x.to(device) for x in X]
            # if events.dim() == 3:
            #     events = events.unsqueeze(1)
            # # ensure masks is [B,1,H,W]
            # if masks.dim() == 3:
            #     masks = masks.unsqueeze(1)

            # events = events.to(device)
            masks = masks.to(device)

            if hidden_state is not None:
                X.append(hidden_state)

            # Forward pass
            # LSTMNetVIT
            preds, hidden_state = model(X)

            # if args.model_type in ['ConvUNet', 'OrigUNet']:
            #     preds, _ = model([events])  # UNet variants take list input; ignore extras
            # else:  # ViT
            #     preds, _ = model([events, None, None, None])

            # Compute loss
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # detach hidden state
            if hidden_state is not None:
                hidden_state = tuple(h.detach() for h in hidden_state)

            epoch_loss += loss.item()
            writer.add_scalar('train/batch_loss', loss.item(), global_step)
            global_step += 1

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch}/{args.epochs} — Loss: {avg_loss:.6f}')
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_hidden_state = None
            for X, masks in val_loader:
                X = [x.to(device) for x in X]
                masks = masks.to(device)

                if val_hidden_state is not None:
                    X.append(val_hidden_state)

                preds, val_hidden_state = model(X)
                loss = criterion(preds, masks)
                val_loss += loss.item()

                if val_hidden_state is not None:
                    val_hidden_state = tuple(h.detach() for h in val_hidden_state)

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('val/epoch_loss', avg_val_loss, epoch)
        print(f'Validation Loss: {avg_val_loss:.6f}')


        # Checkpoint
        if epoch % args.save_freq == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f'epoch_{epoch:04d}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f'Saved checkpoint: {ckpt_path}')

    writer.close()


if __name__ == '__main__':
    main()
