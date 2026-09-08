import torch
import torch.nn as nn
from typing import Tuple

class CNNExtractor(nn.Module):
    def __init__(self, in_channels: int = 1, n_mels: int = 80, cnn_out_channels: int = 64):
        super(CNNExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.Hardtanh(0.0, 20.0, inplace=True)
        self.conv2 = nn.Conv2d(32, cnn_out_channels, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(cnn_out_channels)
        self.act2 = nn.Hardtanh(0.0, 20.0, inplace=True)
        self.freq_dim = (n_mels + 1) // 2
        self.freq_dim = (self.freq_dim + 1) // 2
        self.out_dim = cnn_out_channels * self.freq_dim

    def compute_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        l1 = (input_lengths + 1) // 2
        l2 = (l1 + 1) // 2
        return l2

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        output_lengths = self.compute_output_lengths(input_lengths)
        return x, output_lengths
