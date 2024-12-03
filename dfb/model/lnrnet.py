import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import copy

from typing import Union, Tuple, List

class ConvBnActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, depthwise):
        super(ConvBnActivation, self).__init__()
        self.activation = activation
        self.bn = nn.BatchNorm1d(out_channels)
        padding_size = int((kernel_size-1)/(2))
        if depthwise:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, in_channels, kernel_size, stride=1, padding=padding_size, groups=in_channels),
                torch.nn.BatchNorm1d(in_channels),
                self.activation,
                torch.nn.Conv1d(in_channels, out_channels, 1, stride=1, padding=0)
            )
        else:
            self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding_size)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, activation, depthwise, residual_connection):
        super(ResBlock, self).__init__()
        self.activation = activation
        self.residual_connection = residual_connection

        self.layer1 = ConvBnActivation(in_channels, mid_channels, kernel_size, activation, depthwise)
        self.layer2 = ConvBnActivation(mid_channels, out_channels, kernel_size, activation, depthwise)

        if residual_connection:
            self.skip_conv = torch.nn.Conv1d(in_channels, out_channels, 1, stride=1, padding=0)
        
    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)

        # if self.residual_connection:
        #     y = y + self.skip_conv(x)

        return y

class LNRNet(nn.Module):
    def __init__(self, n_classes: int=10):
        super(LNRNet, self).__init__()

        activation = nn.ReLU()
        kernel_size = 7
        residual_connection = False
        depthwise = True

        self.first_conv = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Conv1d(1, 16, 64, stride=8, padding=28)
        )

        self.conv_layers = nn.Sequential(
            # conv1
            torch.nn.BatchNorm1d(16),
            activation,
            torch.nn.MaxPool1d(2, 2),
            # conv2
            ResBlock(16, 32, 64, kernel_size, activation, depthwise, residual_connection),
            torch.nn.MaxPool1d(2, 2),
            # conv3
            ResBlock(64, 64, 64, kernel_size, activation, depthwise, residual_connection),
            ResBlock(64, 64, 64, kernel_size, activation, depthwise, residual_connection),
            torch.nn.MaxPool1d(2, 2),
            torch.nn.Conv1d(64, 64, 3, stride=1, padding=0),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, 2),
        )

        self.linear_layers = nn.Sequential(
            torch.nn.AdaptiveAvgPool1d((1)),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, n_classes),
        )

    def _fused_model(self):
        qm = copy.deepcopy(self)
        qm = qm.cpu().eval()
        torch.quantization.fuse_modules(qm, ['first_conv.1', 'conv_layers.0', 'conv_layers.1'], inplace=True)

        torch.quantization.fuse_modules(qm, ['conv_layers.3.layer1.conv.0', 'conv_layers.3.layer1.conv.1', 'conv_layers.3.layer1.conv.2'], inplace=True)
        torch.quantization.fuse_modules(qm, ['conv_layers.3.layer1.conv.3', 'conv_layers.3.layer1.bn', 'conv_layers.3.layer1.activation'], inplace=True)

        torch.quantization.fuse_modules(qm, ['conv_layers.3.layer2.conv.0', 'conv_layers.3.layer2.conv.1', 'conv_layers.3.layer2.conv.2'], inplace=True)
        torch.quantization.fuse_modules(qm, ['conv_layers.3.layer2.conv.3', 'conv_layers.3.layer2.bn', 'conv_layers.3.layer2.activation'], inplace=True)

        torch.quantization.fuse_modules(qm, ['conv_layers.5.layer1.conv.0', 'conv_layers.5.layer1.conv.1', 'conv_layers.5.layer1.conv.2'], inplace=True)
        torch.quantization.fuse_modules(qm, ['conv_layers.5.layer1.conv.3', 'conv_layers.5.layer1.bn', 'conv_layers.5.layer1.activation'], inplace=True)

        torch.quantization.fuse_modules(qm, ['conv_layers.5.layer2.conv.0', 'conv_layers.5.layer2.conv.1', 'conv_layers.5.layer2.conv.2'], inplace=True)
        torch.quantization.fuse_modules(qm, ['conv_layers.5.layer2.conv.3', 'conv_layers.5.layer2.bn', 'conv_layers.5.layer2.activation'], inplace=True)

        torch.quantization.fuse_modules(qm, ['conv_layers.6.layer1.conv.0', 'conv_layers.6.layer1.conv.1', 'conv_layers.6.layer1.conv.2'], inplace=True)
        torch.quantization.fuse_modules(qm, ['conv_layers.6.layer1.conv.3', 'conv_layers.6.layer1.bn', 'conv_layers.6.layer1.activation'], inplace=True)

        torch.quantization.fuse_modules(qm, ['conv_layers.6.layer2.conv.0', 'conv_layers.6.layer2.conv.1', 'conv_layers.6.layer2.conv.2'], inplace=True)
        torch.quantization.fuse_modules(qm, ['conv_layers.6.layer2.conv.3', 'conv_layers.6.layer2.bn', 'conv_layers.6.layer2.activation'], inplace=True)

        torch.quantization.fuse_modules(qm, ['conv_layers.8', 'conv_layers.9', 'conv_layers.10'], inplace=True)

        torch.quantization.fuse_modules(qm, ['linear_layers.2', 'linear_layers.3'], inplace=True)

        return qm

    def _get_intermediate_output(self, x:torch.Tensor) -> torch.Tensor:
        x = self.first_conv(x)

        x = self.conv_layers[0](x)
        x = self.conv_layers[1](x)
        x = self.conv_layers[2](x)
        x = self.conv_layers[3](x)
        x = self.conv_layers[4](x)
        out1 = x
        x = self.conv_layers[5](x)
        out2 = x
        x = self.conv_layers[6](x)
        x = self.conv_layers[7](x)
        out3 = x
        x = self.conv_layers[8](x)
        x = self.conv_layers[9](x)
        x = self.conv_layers[10](x)
        x = self.conv_layers[11](x)
        out4 = x

        x = self.linear_layers[0](x)
        x = self.linear_layers[1](x)
        x = self.linear_layers[2](x)

        return out1, out2, out3, out4, x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_conv(x)
        x = self.conv_layers(x)
        x = self.linear_layers(x)

        return x