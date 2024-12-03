import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from typing import Union, Tuple, List

class ConvBnActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBnActivation, self).__init__()
        self.activation = torch.nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        padding_size = int((kernel_size-1)/(2))
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, in_channels, kernel_size, stride=1, padding=padding_size, groups=in_channels),
            torch.nn.BatchNorm1d(in_channels),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels, out_channels, 1, stride=1, padding=0)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size):
        super(ResBlock, self).__init__()
        self.activation = torch.nn.ReLU()
        padding_size = int((kernel_size-1)/(2))

        self.layer1 = ConvBnActivation(in_channels, mid_channels, kernel_size)
        self.layer2 = ConvBnActivation(mid_channels, out_channels, kernel_size)
        self.skip_conv = torch.nn.Conv1d(in_channels, out_channels, 1, stride=1, padding=0)
        
    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)

        y = y + self.skip_conv(x)

        return y

class MyModelEdge(nn.Module):
    def __init__(self, n_classes: int=10, device: str=None, kernel_size=3):
        super(MyModelEdge, self).__init__()

        self.first_conv = torch.nn.Conv1d(1, 16, 64, stride=8, padding=28)

        self.conv_layers = nn.Sequential(
            # conv1
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, 2),
            # conv2
            ResBlock(16, 32, 64, kernel_size),
            torch.nn.MaxPool1d(2, 2),
            # conv3
            ResBlock(64, 64, 64, kernel_size),
            torch.nn.MaxPool1d(2, 2),
            torch.nn.Conv1d(64, 64, 3, stride=1, padding=0),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, 2),
        )

        with torch.no_grad():
            dummy = torch.rand(1, 1, 2048).to(device)
            dummy = self.first_conv.to(device)(dummy)
            dummy = self.conv_layers.to(device)(dummy)
            dummy = torch.flatten(dummy, 1)
            lin_input = dummy.shape[1]

        self.linear_layers = nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(lin_input, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_conv(x)
        x = self.conv_layers(x)
        x = self.linear_layers(x)

        return x