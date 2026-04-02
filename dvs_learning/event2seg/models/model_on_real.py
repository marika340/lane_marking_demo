"""
@authors: A Bhattacharya, et. al
@organization: GRASP Lab, University of Pennsylvania
@date: ...
@license: ...

@brief: This module contains the models that were used in the paper "Utilizing vision transformer models for end-to-end vision-based
quadrotor obstacle avoidance" by Bhattacharya, et. al
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
import torch.nn.utils.spectral_norm as spectral_norm
from ViTsubmodules import *

# def refine_inputs(X):
#     # fill quaternion rotation if not given
#     # make it [1, 0, 0, 0] repeated with numrows = X[0].shape[0]
#     # print(f"X[0] size: {X[0].shape}")
#     # print(f"Before: X[2] size: {X[2].shape}")
#     if X[2] is None:
#         # X[2] = torch.Tensor([1, 0, 0, 0]).float()
#         X[2] = torch.zeros((X[0].shape[0], 4)).float().to(X[0].device)
#         X[2][:, 0] = 1
#     # print(f"After: X[2] size: {X[2].shape}")
#     # if input depth images are not of right shape, resize
#     if X[0].shape[-2] != 60 or X[0].shape[-1] != 90:
#         X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear')

#     return X

# def refine_inputs(X):
#     """
#     X: list where X[0] is segmentation‐mask tensor of shape (B,1,H,W).
#     We downsample to (60,90) for the ViT’s patch embedding.
#     """
#     mask = X[0]
#     # enforce 4‑D:  (B,1,H,W) → (B,1,60,80)
#     # if mask.dim()==3:  
#     #     mask = mask.unsqueeze(1)   # just in case someone passed (B,H,W)
#     if mask.dim()==3:
#         if mask.shape[0] == 1:
#             # (C,H,W) → (1,C,H,W)
#             mask = mask.unsqueeze(0)
#         else:
#             # (B,H,W) → (B,1,H,W)
#             mask = mask.unsqueeze(1)
#     elif mask.dim() == 2:
#         # (H,W) → (1,1,H,W)
#         mask = mask.unsqueeze(0).unsqueeze(0)
#     mask = F.interpolate(mask, size=(60, 90), mode='bilinear', align_corners=False)
#     return [mask]

# def refine_inputs(X):
#     """
#     Ensure the mask input has shape (B, 1, H, W) before resizing to (60, 90).
#     """
#     mask = X[0]

#     # Handle 2D: (H, W) → (1,1,H,W)
#     if mask.dim() == 2:
#         mask = mask.unsqueeze(0).unsqueeze(0)

#     # Handle 3D: could be (B,H,W) or (C,H,W)
#     elif mask.dim() == 3:
#         if mask.shape[0] in [1, 3]:  # assume it's (C,H,W)
#             mask = mask.unsqueeze(0)
#         else:  # assume it's (B,H,W)
#             mask = mask.unsqueeze(1)

#     # If not 4D now, something's wrong
#     if mask.dim() != 4:
#         raise ValueError(f"Expected 4D input for interpolate, got shape: {mask.shape}")

#     # Resize
#     mask = F.interpolate(mask, size=(60, 90), mode='bilinear', align_corners=False)

#     return [mask]

def refine_inputs(X):
    """
    X: list with X[0] = mask tensor, shape (B, T, C, H, W)
    Returns downsampled mask of shape (B, T, 1, 60, 90)
    """
    mask = X[0]
    # print(f"mask.shape = {mask.shape}")
    if mask.dim() == 5:
        if mask.shape[-2] != 60 or mask.shape[-1] != 90:
            B, T, C, H, W = mask.shape
            mask = mask.view(B*T, C, H, W)  # (B*T, C, H, W)
            mask = F.interpolate(mask, size=(60, 90), mode='bilinear', align_corners=False)
            mask = mask.view(B, T, C, 60, 90)
            mask = mask.unsqueeze(0)  # (1, B, T, 1, 60, 90)
    else:
        if mask.shape[-2] != 60 or mask.shape[-1] != 90:
            mask = F.interpolate(mask, size=(60, 90), mode='bilinear', align_corners=False)
            
    # print(f"[refine_inputs] mask shape before binarization: {mask.shape}, unique vals: {torch.unique(mask)}")
    mask = torch.where(mask > 0.08, 1.0, 0.0) # Binarize mask
    # print(f"[refine_inputs] mask shape after binarization: {mask.shape}, unique vals: {torch.unique(mask)}")
    
    return [mask]

# def refine_inputs_vitlstm(X, out_hw=(60, 90)):
#     """
#     X[0]: mask sequence, expected shape (B, T, C, H, W).
#     Returns: [mask] with shape (B, T, C, out_hw[0], out_hw[1])
#     """
#     mask = X[0]
#     if mask.dim() != 5:
#         raise ValueError(f"Expected (B,T,C,H,W); got {tuple(mask.shape)}")
#     B, T, C, H, W = mask.shape
#     mask = mask.view(B*T, C, H, W)
#     mask = F.interpolate(mask, size=out_hw, mode='bilinear', align_corners=False)
#     mask = mask.view(B, T, C, out_hw[0], out_hw[1])
#     return [mask]

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, output_dim=2, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64, 128)         # after global avg pool -> 64 features
        self.fc2 = nn.Linear(128, output_dim)

    def _ensure_tensor4d(self, x):
        # If list/tuple, take the image tensor as first element
        if isinstance(x, (list, tuple)):
            x = x[0]
        # If sequence input, take first frame (B,T,C,H,W) -> (B,C,H,W)
        if x.dim() == 5:
            x = x[:, 0]  # quick baseline: first frame only
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B,C,H,W), got shape {tuple(x.shape)}")
        return x

    def forward(self, x):
        # x = self._ensure_tensor4d(x)
        # Handle (B,T,C,H,W) or (B,C,H,W)
        if isinstance(x, (list, tuple)):
            x = x[0]
        
        # is_seq = False
        # if x.dim() == 5:  # (B,T,C,H,W)
        #     B, T, C, H, W = x.shape
        #     x = x.view(B*T, C, H, W)
        #     is_seq = True
        # elif x.dim() != 4:
        #     raise ValueError(f"Expected 4D or 5D tensor, got shape {tuple(x.shape)}")

        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B,C,H,W), got shape {tuple(x.shape)}")
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool2d(x, 1)  # (B,64,1,1)
        x = x.view(x.size(0), -1)        # (B,64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                # (B, output_dim)
        
        # If sequence, average over time to get per-batch output
        # if is_seq:
        #     x = x.view(B, T, -1).mean(dim=1)
            
        return [x], None

class ConvNet(nn.Module):
    """
    Conv + FC Network 
    Num Params: 235,269
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, 3)
        self.conv2 = nn.Conv2d(4, 10, 3, 2)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.maxpool = nn.MaxPool2d(2, 1)
        self.bn1 = nn.BatchNorm2d(4)
        
        self.fc0 = nn.Linear(845, 256, bias=False)
        self.fc1 = nn.Linear(256, 64, bias=False)
        self.fc2 = nn.Linear(64, 32, bias=False)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, X):

        X = refine_inputs(X)

        x = X[0]
        x = -self.maxpool(- self.bn1(F.relu(self.conv1(x))))
        x = self.avgpool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        metadata = torch.cat((X[1]*0.1, X[2]), dim=1).float()

        x = torch.cat((x, metadata), dim=1).float()

        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)

        return x, None #None is passed to be compatible with hidden dimensions

class LSTMNet(nn.Module):
    """
    LSTM + FC Network 
    Num Params: 2,949,937
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, stride = 3, padding=1)
        self.conv2 = nn.Conv2d(4, 10, 3,stride =  2, padding=0)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.maxpool = nn.MaxPool2d(3, 1)
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(10)

        self.lstm = LSTM(input_size=665, hidden_size=395,
                         num_layers=2, dropout=0.15, bias=False)
        self.fc1 = spectral_norm(nn.Linear(395, 64))
        self.fc2 = spectral_norm(nn.Linear(64, 16))
        self.fc3 = spectral_norm(nn.Linear(16, 3))

    def forward(self, X):

        X = refine_inputs(X)

        x = X[0]
        x = -self.maxpool(-self.bn1(F.relu(self.conv1(x))))
        x = self.avgpool(self.bn2(F.relu(self.conv2(x))))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.cat((x,X[1]*0.1, X[2]), dim=1).float()
        if len(X)>3:
            x,h = self.lstm(x, X[3])
        else:
            x,h = self.lstm(x)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x, h

class LSTMNetVIT(nn.Module):
    def __init__(self, output_dim=5, return_sequence=False):
        super().__init__()
        self.return_sequence = return_sequence

        # 2-stage MixTransformer encoder (same as original)
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(1, 32,
                patch_size=7, stride=4, padding=3,
                n_layers=2, reduction_ratio=8,
                num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64,
                patch_size=3, stride=2, padding=1,
                n_layers=2, reduction_ratio=4,
                num_heads=2, expansion_factor=8)
        ])
        # project flattened features to 512
        self.decoder = spectral_norm(nn.Linear(4608, 512))
        # LSTM to capture temporal dynamics over feature vector
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=3,
            dropout=0.1,
            batch_first=False  # we'll feed as (seq_len=1, batch, 512)
        )
        # final head -> 5 AckermannDrive dims: [steering, steer_vel, speed, accel, jerk]
        self.nn_fc2 = spectral_norm(nn.Linear(128, output_dim))

        # helper layers for spatial reconstruction
        self.up_sample   = nn.Upsample(size=(16, 24), mode='bilinear', align_corners=True)
        self.pxShuffle   = nn.PixelShuffle(upscale_factor=2)
        self.down_sample = nn.Conv2d(48, 12, kernel_size=3, padding=1)

        self.head = spectral_norm(nn.Linear(128, output_dim))

    def forward(self, X, hidden=None):
        """
        X: [ mask_seq ] → shape: (B, T, 1, 60, 90)
        """
        if isinstance(X, list) or isinstance(X, tuple):
            X = X[0]  # unwrap
            # print(f"[DEBUG] Unwrapped input X shape: {X.shape}") # [8, 16, 1, 60, 90]
            if X.dim() == 4:
                X = X.unsqueeze(1)  # (B, 1, C, H, W) if single frame
            # X = X.unsqueeze(1)  # (B, 1, C, H, W)
        # print(f"[DEBUG] Original input X shape: {X.shape}") # [8, 1, 16, 1, 60, 90]
        # X = refine_inputs(X)
        # x = X[0]  # (B, T, 1, 60, 90)
        x = X

        B, T, C, H, W = x.shape
        # print(f"[DEBUG] Input x shape: {x.shape}")

        # Flatten batch and time for ViT
        x = x.view(B * T, C, H, W)
        
        # Run through encoder blocks
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))
        out0, out1 = embeds[1], embeds[2]

        up = self.up_sample(out0)       # (B,32,16,24)
        # ps = self.pxShuffle(out1)       # (B,12,16,24)
        # ps = ps.transpose(2, 3)  # Swap height and width: (B, C, 24, 16) → (B, C, 16, 24)

        # Assuming input mask size = (B, T, 1, 90, 60)
        # So per-frame: (B, 1, 90, 60) → patch processing shrinks and increases channels
        # print(f"[DEBUG] out1 shape before pxShuffle: {out1.shape}")
        ps = self.pxShuffle(out1)  # Output: (B, C*4, H//2, W//2)
        # print(f"[DEBUG] After pxShuffle: {ps.shape}")

        # Calculate expected spatial dimensions after reshaping
        # patch_h, patch_w = 5, 5  # depends on ViT layer config
        # input_h, input_w = 90, 60
        # H, W = input_h // patch_h, input_w // patch_w  # (18, 12) if patch_h=5

        # Reshape to match spatial shape of up
        # ps = ps.transpose(1, 2).contiguous().view(ps.size(0), -1, H, W)  # (B, C, H, W)
        # ps = ps.transpose(2, 3)
        # print(f"[DEBUG] after transpose: {ps.shape}")
        # print(f"[DEBUG] up shape: {up.shape}")
        # print(f"up shape: {up.shape}, ps shape: {ps.shape}")
        cat = torch.cat([up, ps], dim=1)  # (B,44,16,24)
        cat = self.down_sample(cat)     # (B,12,16,24)

        flat = cat.flatten(1)           # (B,4608)
        dec = self.decoder(flat)        # (B,512)

        # Reshape for LSTM: (T, B, 512)
        dec_seq = dec.view(B, T, -1).permute(1, 0, 2)

        # LSTM
        if hidden is None:
            out_seq, h = self.lstm(dec_seq)
        else:
            out_seq, h = self.lstm(dec_seq, hidden)

        if self.return_sequence:
            # (T, B, 128) -> (T, B, output_dim) -> (B, T, output_dim)
            pred_tb = self.head(out_seq)
            pred_bt = pred_tb.permute(1, 0, 2)  # (B, T, output_dim)
            return pred_bt, h
        
        else:
            # last timestep only: (B, 128) -> (B, output_dim)
            last = out_seq[-1]  # (B, 128)
            pred_b = self.head(last)
            return pred_b, h
        # train
        # out = out_seq[-1]
        # out = self.nn_fc2(out) # (B, 5)
        # evaluate
        # out = self.nn_fc2(out_seq)  # (T, B, 128) → (T, B, 5)
        # out = out.permute(1, 0, 2)  # (T, B, 5) → (B, T, 5)

        # return out, h
    

class ViT(nn.Module):
    """
    Seg‑Mask → [steering, acceleration]
    """
    def __init__(self, output_dim=2, return_sequence=False):
        super().__init__()
        self.return_sequence = return_sequence
        self.output_dim = output_dim
        # keep the same encoder blocks:
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(1, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8)
        ])  
        # self.decoder = nn.Linear(4608, 512)
        # self.nn_fc1 = spectral_norm(nn.Linear(512, 256))
        # self.nn_fc2 = spectral_norm(nn.Linear(256, output_dim))
        self.up_sample = nn.Upsample(size=(16,24), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.down_sample = nn.Conv2d(48,12,3, padding = 1)

        self.cnn_head = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),

            # collapse to 1 channel
            nn.Conv2d(16, 1, kernel_size=1)
        )


    def forward(self, X):

        X = refine_inputs(X)

        x = X[0]
        # x = X
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))        
        out = embeds[1:]
        out = torch.cat([self.pxShuffle(out[1]),self.up_sample(out[0])],dim=1) # (B,48,16,24)
        # print(f"[ViT] concatenated feature map shape: {out.shape}")
        # CNN with output shape (B,1,20,30)
        out = self.cnn_head(out)
        out = F.interpolate(out, size=(60, 90), mode='bilinear', align_corners=True)
        # Use sigmoid if we use MSE loss as loss, but do NOT use sigmoid if we use BCEWithLogitsLoss, because that loss already applies sigmoid internally.
        # out = torch.sigmoid(out)
        # out = self.down_sample(out)
        # out = self.decoder(out.flatten(1))
        # I don't need this, because I don't have desvel and currquat in the input X
        # out = torch.cat([out, X[1]/10, X[2]], dim=1).float()
        # out = F.leaky_relu(self.nn_fc1(out))
        # out = self.nn_fc2(out)

        return out, None


class UNetConvLSTMNet(nn.Module):
    """
    UNet+LSTM Network 
    Num Params: 2,955,822 
    """

    def __init__(self):
        super().__init__()

        self.unet_e11 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.unet_e12 = nn.Conv2d(4, 4, kernel_size=3, padding=1) #(N, 4, 60, 90)
        self.unet_pool1 = nn.MaxPool2d(kernel_size=2, stride=3,) #(N, 4, 30, 45)

        self.unet_e21 = nn.Conv2d(4, 8, kernel_size=3, padding=1) #(N, 8, 26, 41)
        self.unet_e22 = nn.Conv2d(8, 8, kernel_size=3, padding=1) #(N, 8, 24, 39)
        self.unet_pool2 = nn.MaxPool2d(kernel_size=2, stride=2,) #(N, 8, 12, 19)

        #Input: (N, 8, 12, 19)
        self.unet_e31 = nn.Conv2d(8, 16, kernel_size=3, padding=1) #(N, 8, 10, 17)
        self.unet_e32 = nn.Conv2d(16, 16, kernel_size=3, padding=1) #(N, 16, 8, 15)

        self.unet_upconv1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2,)
        self.unet_d11 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.unet_d12 = nn.Conv2d(8, 8, kernel_size=3, padding=1)

        self.unet_upconv2 = nn.ConvTranspose2d(8, 4, kernel_size=3, stride=3,)
        self.unet_d21 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.unet_d22 = nn.Conv2d(4, 4, kernel_size=3, padding=1)

        self.unet_out = nn.Conv2d(4, 1, kernel_size=1)

        self.conv_conv1 = nn.Conv2d(2, 4, 5, 3)
        self.conv_conv2 = nn.Conv2d(4, 10, 5, 2)
        self.conv_avgpool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.conv_maxpool = nn.MaxPool2d(2, 1)
        self.conv_bn1 = nn.BatchNorm2d(4)

        self.lstm = LSTM(input_size=3065, hidden_size=200, num_layers=2, dropout=0.15, bias=False)

        self.nn_fc1 = torch.nn.utils.spectral_norm(nn.Linear(200, 64))
        self.nn_fc2 = torch.nn.utils.spectral_norm(nn.Linear(64, 32))
        self.nn_fc3 = torch.nn.utils.spectral_norm(nn.Linear(32, 3))

    def forward(self, X):

        X = refine_inputs(X)

        img, des_vel, quat = X[0], X[1], X[2]
        y_e1 = torch.relu(self.unet_e12(torch.relu(self.unet_e11(img))))
        unet_enc1 = self.unet_pool1(y_e1)
        y_e2 = torch.relu(self.unet_e22(torch.relu(self.unet_e21(unet_enc1))))
        unet_enc2 = self.unet_pool2(y_e2)
        y_e3 = torch.relu(self.unet_e32(torch.relu(self.unet_e31(unet_enc2))))

        unet_dec1 = torch.relu(self.unet_d12(torch.relu(self.unet_d11(torch.cat([self.unet_upconv1(y_e3), y_e2], dim=1)))))
        unet_dec2 = torch.relu(self.unet_d22(torch.relu(self.unet_d21(torch.cat([self.unet_upconv2(unet_dec1), y_e1], dim=1)))))

        y_unet = self.unet_out(unet_dec2)
        x_conv = torch.cat((img, y_unet), dim=1)

        y_conv = -self.conv_maxpool(-torch.relu(self.conv_bn1(self.conv_conv1(x_conv))))
        y_conv = self.conv_avgpool(torch.relu(self.conv_conv2(y_conv)))

        x_lstm = torch.cat([torch.flatten(y_conv, 1), torch.flatten(y_e3, 1), des_vel*0.1, quat], dim=1).float()

        if len(X)>3:
            y_lstm, h = self.lstm(x_lstm, X[3])
        else:
            y_lstm, h = self.lstm(x_lstm)

    
        y_fc1 = F.leaky_relu(self.nn_fc1(y_lstm))
        y_fc2 = F.leaky_relu(self.nn_fc2(y_fc1))
        y = self.nn_fc3(y_fc2)

        return y, h

if __name__ == '__main__':
    print("MODEL NUM PARAMS ARE")
    model = ConvNet().float()
    print("ConvNet: ")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = LSTMNet().float()
    print("LSTMNet: ")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = UNetConvLSTMNet().float()
    print("UNET: ")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = ViT().float()
    print("VIT: ")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = LSTMNetVIT().float()
    print("VITLSTM: ")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
