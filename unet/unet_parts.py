import torch.nn as nn
import torch.nn.functional as F

from torch import cat


class DoubleConv(nn.Module):
    # L_out = (L_in + 2*padding - dilation*(kernel_size - 1)-1)/stride + 1
    # Solve for padding
    def __init__(self, inc, outc):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(inc, outc, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(outc),
            nn.Conv1d(outc, outc, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(outc),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, inc, outc, size=4):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=size), DoubleConv(inc, outc)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    # L_out = (L_in - 1) * stride - 2 * padding + dilation*(kernel_size - 1) + output_padding + 1
    # Solve for padding
    """Upscaling then double conv"""

    def __init__(self, inc, outc, setting=4):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            inc, outc, kernel_size=setting, stride=setting)
        self.conv = DoubleConv(2 * outc, outc)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = x2.shape[2] - x1.shape[2]
        x1 = F.pad(x1, (diff // 2, diff - diff // 2))
        x = cat([x1, x2], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, inc, outc):
        super(OutConv, self).__init__()
        self.conv_sigmoid = nn.Sequential(
            nn.Conv1d(inc, outc, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv_sigmoid(x)
