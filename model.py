"""
DeepfakeDetector - Dual-Branch Architecture
Branch 1: EfficientNet-B0 for spatial/RGB feature extraction
Branch 2: Frequency-domain branch (FFT artifacts)
Fused with Channel Attention for interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ─────────────────────────────────────────────────────────────
# Channel Attention (Squeeze-and-Excitation)
# ─────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))

        scale = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * scale


# ─────────────────────────────────────────────────────────────
# Frequency Branch (FFT-based)
# ─────────────────────────────────────────────────────────────
class FrequencyBranch(nn.Module):
    def __init__(self, out_features=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.fc = nn.Linear(128 * 4 * 4, out_features)

    def forward(self, x):
        # Convert RGB → grayscale
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]

        # FFT processing
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.log(torch.abs(fft_shift) + 1e-8)

        # Normalize to [0,1]
        min_val = magnitude.amin(dim=(-2, -1), keepdim=True)
        max_val = magnitude.amax(dim=(-2, -1), keepdim=True)
        magnitude = (magnitude - min_val) / (max_val - min_val + 1e-8)

        magnitude = magnitude.unsqueeze(1)  # (B,1,H,W)

        feat = self.conv(magnitude)
        feat = feat.flatten(1)

        return self.fc(feat)


# ─────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────
class DeepfakeDetector(nn.Module):
    def __init__(self, freq_features=128, dropout=0.4):
        super().__init__()

        # ── RGB Branch (EfficientNet-B0) ──
        backbone = models.efficientnet_b0(weights="DEFAULT")

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        rgb_dim = backbone.classifier[1].in_features  # 1280

        # ✅ FIX: Use correct channel size
        self.channel_attention = ChannelAttention(in_channels=rgb_dim)

        # ── Frequency Branch ──
        self.freq_branch = FrequencyBranch(out_features=freq_features)

        # ── Fusion ──
        fused_dim = rgb_dim + freq_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),
        )

        # Grad-CAM target
        self.gradcam_target = self.features[-1]

    def forward(self, x):
        # RGB branch
        rgb_feat = self.features(x)                 # (B,1280,7,7)
        rgb_feat = self.channel_attention(rgb_feat)
        rgb_feat = self.avgpool(rgb_feat).flatten(1)  # (B,1280)

        # Frequency branch
        freq_feat = self.freq_branch(x)            # (B,128)

        # Fusion
        fused = torch.cat([rgb_feat, freq_feat], dim=1)

        return self.classifier(fused)

    def get_last_conv_layer(self):
        return self.gradcam_target


# ─────────────────────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = DeepfakeDetector()

    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)

    print(f"Output shape: {out.shape}")  # Expected: (2, 1)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")