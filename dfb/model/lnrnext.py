import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from typing import Union, Tuple, List

class ConvBnRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, groups):
        super(ConvBnRelu, self).__init__()

        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class SeparableConv1D(torch.nn.Module):
    def __init__(self, input_length, in_channels, out_channels, kernel_size):
        super(SeparableConv1D, self).__init__()

        padding = int((kernel_size-1)/(2))

        self.dwconv = torch.nn.Conv1d(in_channels, in_channels, kernel_size, 1, padding, 1, in_channels)
        self.ln = torch.nn.LayerNorm((in_channels, input_length))
        self.pwconv = torch.nn.Conv1d(in_channels, out_channels, 1, 1, 0, 1, 1)
        self.gelu = torch.nn.GELU()
    
    def forward(self, x):
        x = self.dwconv(x)
        x = self.ln(x)
        x = self.pwconv(x)
        x = self.gelu(x)

        return x

class ResBlock(nn.Module):
    def __init__(self, input_length, in_channels, mid_channels, out_channels, kernel_size, residual_connection=False):
        super(ResBlock, self).__init__()

        self.identical = in_channels == out_channels
        self.residual = residual_connection

        self.layer1 = SeparableConv1D(input_length, in_channels, mid_channels, kernel_size)
        self.layer2 = SeparableConv1D(input_length, mid_channels, out_channels, kernel_size)

        self.add_relu = torch.nn.quantized.FloatFunctional()
        
    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)

        if self.identical and self.residual:
            y = self.add_relu.add(x, y)

        return y

class LnrNext(nn.Module):
    def __init__(self, n_classes: int=10):
        super(LnrNext, self).__init__()

        self.dropout = torch.nn.Dropout(0.5)
        self.conv1 = ConvBnRelu(1, 16, 64, 28, 8, 1)
        self.pool1 = torch.nn.MaxPool1d(2, 2)
        self.block1 = ResBlock(128, 16, 32, 64, 7, True)
        self.pool2 = torch.nn.MaxPool1d(2, 2)
        self.block2 = ResBlock(64, 64, 64, 64, 7, True)
        self.block3 = ResBlock(64, 64, 64, 64, 7, True)
        self.pool3 = torch.nn.MaxPool1d(2, 2)
        self.conv2 = ConvBnRelu(64, 64, 3, 1, 1, 1)

        self.lin1 = torch.nn.Linear(64, 100)
        self.bn1 = torch.nn.BatchNorm1d(100)
        self.relu1 = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(100, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.block1(x)
        x = self.pool2(x)

        x = self.block2(x)
        x = self.block3(x)
        x = self.pool3(x)

        x = self.conv2(x)

        x = x.mean(2)
        x = self.lin1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.lin2(x)

        return x