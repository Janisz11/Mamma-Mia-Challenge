import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class SpatialAttention3D(nn.Module):
    """CBAM-style spatial attention for 3-D feature maps."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        p = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=p, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        att = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att  # [B, C, D, H, W]


class ResNet3D_CBAM(nn.Module):
    """Pre-trained R3D-18 with spatial attention on layers 2-4.

    The first conv is adapted to accept `in_channels` input channels by
    copying pretrained weights for the first 3 channels and initialising
    the remaining channels from the mean of the pretrained weights.
    """

    def __init__(self, in_channels: int = 6, pretrained: bool = True,
                 freeze_until: str | None = "layer1"):
        super().__init__()

        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        net = r3d_18(weights=weights)

        old_conv = net.stem[0]
        net.stem[0] = nn.Conv3d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                net.stem[0].weight[:, :3] = old_conv.weight
                if in_channels > 3:
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                    net.stem[0].weight[:, 3:] = mean_w.repeat(1, in_channels - 3, 1, 1, 1)

        if freeze_until is not None:
            for name, p in net.named_parameters():
                if not name.startswith(freeze_until):
                    p.requires_grad = False

        self.stem = net.stem
        self.layer1 = net.layer1
        self.layer2 = nn.Sequential(net.layer2, SpatialAttention3D())
        self.layer3 = nn.Sequential(net.layer3, SpatialAttention3D())
        self.layer4 = nn.Sequential(net.layer4, SpatialAttention3D())
        self.avgpool = net.avgpool

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)   # [B, 512, 1, 1, 1]
        return self.proj(x)   # [B, 128]


class FusionPCRNet(nn.Module):
    """Multimodal fusion of 3-D DCE-MRI features and TIC-derived statistics.

    Architecture:
        CNN branch   : ResNet3D_CBAM → 128-dim embedding
        TIC branch   : FC network     →  64-dim embedding
        Fusion head  : concat(128+64) → 128 → 2-class logits
    """

    def __init__(self, img_channels: int = 6, tic_dim: int = 4):
        super().__init__()

        self.cnn = ResNet3D_CBAM(img_channels)

        # LayerNorm instead of BatchNorm for batch-size independence
        self.fc_tic = nn.Sequential(
            nn.Linear(tic_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x_img: torch.Tensor, x_tic: torch.Tensor) -> torch.Tensor:
        img_vec = self.cnn(x_img)       # [B, 128]
        tic_vec = self.fc_tic(x_tic)    # [B, 64]
        x = torch.cat([img_vec, tic_vec], dim=1)
        return self.head(x)             # [B, 2]
