 
import torch
import torch.nn as nn
import math

class SpectralSpatialBlock(nn.Module):
    def __init__(self, in_ch=1, bands=270, c3d1=8, c3d2=16, c2d=64):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(in_ch, c3d1, kernel_size=(7,3,3), padding=(3,1,1))
        self.bn3d_1 = nn.BatchNorm3d(c3d1)
        self.conv3d_2 = nn.Conv3d(c3d1, c3d2, kernel_size=(5,3,3), stride=(2,1,1), padding=(2,1,1))
        self.bn3d_2 = nn.BatchNorm3d(c3d2)
        self.act = nn.ReLU(inplace=True)
        self.bands_out = math.ceil(bands / 2)
        self.c2d_in = c3d2 * self.bands_out
        self.conv2d = nn.Conv2d(self.c2d_in, 64, kernel_size=3, padding=1)
        self.bn2d = nn.BatchNorm2d(64)
    def forward(self, x):
        x = self.act(self.bn3d_1(self.conv3d_1(x)))
        x = self.act(self.bn3d_2(self.conv3d_2(x)))
        N, C, B2, P, _ = x.shape
        x = x.view(N, C*B2, P, P)
        x = self.act(self.bn2d(self.conv2d(x)))
        return x

class LocalGlobalFusion(nn.Module):
    def __init__(self, channels=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels*4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(channels*4, channels), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(channels)
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        N, C, P, _ = x.shape
        seq = x.flatten(2).transpose(1, 2)
        res = seq
        seq = self.norm1(seq)
        attn_out, _ = self.mhsa(seq, seq, seq)
        seq = res + attn_out
        res = seq
        seq = self.norm2(seq)
        seq = res + self.ffn(seq)
        x_global = seq.transpose(1, 2).view(N, C, P, P)
        x_local = self.local_branch(x)
        return x_global + x_local

class MultiOutputHeads(nn.Module):
    def __init__(self, in_ch=64, num_classes=9):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head1 = nn.Linear(in_ch, num_classes)
        self.head2 = nn.Linear(in_ch, num_classes)
        self.head3 = nn.Linear(in_ch, num_classes)
    def forward(self, f1, f2, f3):
        def head(feat, linear):
            return linear(self.pool(feat).flatten(1))
        return head(f1, self.head1), head(f2, self.head2), head(f3, self.head3)

class CropHSINet(nn.Module):
    def __init__(self, bands=270, num_classes=9, heads=4, c2d=64):
        super().__init__()
        self.backbone = SpectralSpatialBlock(in_ch=1, bands=bands, c2d=c2d)
        self.lg1 = LocalGlobalFusion(channels=c2d, num_heads=heads)
        self.lg2 = LocalGlobalFusion(channels=c2d, num_heads=heads)
        self.heads = MultiOutputHeads(in_ch=c2d, num_classes=num_classes)
    def forward(self, x):
        f1 = self.backbone(x)
        f2 = self.lg1(f1)
        f3 = self.lg2(f2)
        return self.heads(f1, f2, f3)
