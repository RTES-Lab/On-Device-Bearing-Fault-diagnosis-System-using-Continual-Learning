import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from typing import Union, Tuple, List

class SpatialAttention(torch.nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = int((kernel_size-1)/(2))

        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x = self.conv(torch.cat((avg_out, max_out), dim=1))
        return self.sigmoid(x)

class ChannelAttention(torch.nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels//reduction_ratio, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out  = self.lin(torch.mean(x, dim=2, keepdim=False))
        max_out  = self.lin(torch.max(x, dim=2, keepdim=False)[0])
        return self.sigmoid(torch.unsqueeze(avg_out + max_out, -1))

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

class Dense1D(torch.nn.Module):
    def __init__(self, in_channel, growth_rate, kernel_size):
        super(Dense1D, self).__init__()

        padding = int((kernel_size-1)/(2))

        self.conv1 = ConvBnRelu(in_channel, growth_rate, 1, 0, 1, 1)
        self.conv2 = ConvBnRelu(growth_rate, growth_rate, kernel_size, padding, 1, growth_rate)
        self.conv3 = ConvBnRelu(growth_rate, growth_rate, 1, 0, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x

class SeparableConv1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SeparableConv1D, self).__init__()

        padding = int((kernel_size-1)/(2))

        self.dwconv = ConvBnRelu(in_channels, in_channels, kernel_size, padding, 1, in_channels)
        self.pwconv = ConvBnRelu(in_channels, out_channels, 1, 0, 1, 1)
    
    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)

        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, residual_connection=False, cbam=False):
        super(ResBlock, self).__init__()

        self.identical = in_channels == out_channels
        self.residual = residual_connection

        self.layer1 = SeparableConv1D(in_channels, mid_channels, kernel_size)
        self.layer2 = SeparableConv1D(mid_channels, out_channels, kernel_size)

        self.cbam = cbam

        if cbam:
            self.ca = ChannelAttention(out_channels)
            self.sa = SpatialAttention()
        
    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)

        if self.identical and self.residual:
            y = x + y
        
        if self.cbam:
            y = self.ca(y) * y
            y = self.sa(y) * y

        return y

class CatLnr(nn.Module):
    def __init__(self, n_classes: int=10):
        super(CatLnr, self).__init__()

        # V2
        # self.dropout = torch.nn.Dropout(0.5)
        # self.conv1 = ConvBnRelu(1, 16, 64, 28, 8, 1)
        # self.pool1 = torch.nn.MaxPool1d(2, 2)
        # self.block1 = ResBlock(16, 32, 64, 7, False)
        # self.pool2 = torch.nn.MaxPool1d(2, 2)
        # self.block2 = ResBlock(64, 64, 64, 7, False)
        # self.block3 = ResBlock(64, 64, 64, 7, False)
        # self.pool3 = torch.nn.MaxPool1d(2, 2)
        # self.conv2 = ConvBnRelu(64, 64, 3, 1, 1, 1)

        # self.lin1 = torch.nn.Linear(64*4, 100)
        # self.bn1 = torch.nn.BatchNorm1d(100)
        # self.relu1 = torch.nn.ReLU()
        # self.lin2 = torch.nn.Linear(100, n_classes)

        self.dropout = torch.nn.Dropout(0.5)
        self.conv1 = ConvBnRelu(1, 16, 64, 28, 8, 1)
        self.pool1 = torch.nn.MaxPool1d(2, 2)
        self.block1 = ResBlock(16, 32, 64, 7, False)
        self.pool2 = torch.nn.MaxPool1d(2, 2)
        self.block2 = ResBlock(64, 64, 64, 7, False)
        self.block3 = ResBlock(64, 64, 64, 7, False)
        self.pool3 = torch.nn.MaxPool1d(2, 2)
        self.conv2 = ConvBnRelu(64, 64, 3, 1, 1, 1)

        self.lin1 = torch.nn.Linear(64*4, 100)
        self.bn1 = torch.nn.BatchNorm1d(100)
        self.relu1 = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(100, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.pool1(x)
        out1 = self.block1(x)
        out1 = self.pool2(out1)

        out2 = self.block2(out1)

        out3 = self.block3(out2)
        out3 = self.pool3(out3)

        out4 = self.conv2(out3)

        y = torch.cat((out1.mean(2), out2.mean(2), out3.mean(2), out4.mean(2)), 1)
        y = self.lin1(y)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.lin2(y)

        return y

class CatLnrV3(nn.Module):
    def __init__(self, n_classes: int=10):
        super(CatLnrV3, self).__init__()

        self.dropout = torch.nn.Dropout(0.5)
        self.conv1 = ConvBnRelu(1, 16, 64, 28, 8, 1)
        self.pool1 = torch.nn.MaxPool1d(2, 2)
        self.block1 = ResBlock(16, 32, 64, 7, False)
        self.pool2 = torch.nn.MaxPool1d(2, 2)
        self.block2 = ResBlock(64, 64, 64, 7, False)
        self.block3 = ResBlock(64, 64, 64, 7, False)
        self.pool3 = torch.nn.MaxPool1d(2, 2)
        self.conv2 = ResBlock(64*3, 64*2, 64, 7, False)

        self.lin1 = torch.nn.Linear(64, 100)
        self.bn1 = torch.nn.BatchNorm1d(100)
        self.relu1 = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(100, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.pool1(x)

        out1 = self.block1(x)
        out1 = self.pool2(out1)

        out2 = self.block2(out1)

        out3 = self.block3(out2)
        out3 = self.pool3(out3)

        out1 = self.pool3(out1)
        out2 = self.pool3(out2)

        y = torch.cat((out1, out2, out3), 1)

        y = self.conv2(y)

        y = y.mean(2)
        y = self.lin1(y)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.lin2(y)

        return y


class CatLnrV4(nn.Module):
    def __init__(self, n_classes: int=10):
        super(CatLnrV4, self).__init__()

        self.dropout = torch.nn.Dropout(0.5)
        self.conv1 = ConvBnRelu(1, 16, 64, 28, 8, 1)
        self.pool1 = torch.nn.MaxPool1d(2, 2)
        self.block1 = ResBlock(16, 32, 64, 7, False)
        self.pool2 = torch.nn.MaxPool1d(2, 2)
        self.block2 = ResBlock(64, 64, 64, 7, False)
        self.block3 = ResBlock(64, 64, 64, 7, False)
        self.pool3 = torch.nn.MaxPool1d(2, 2)
        self.squeezer =  ConvBnRelu(64*3, 64, 1, 0, 1, 1)
        self.conv2 = SeparableConv1D(64, 64, 7)

        self.lin1 = torch.nn.Linear(64, 100)
        self.bn1 = torch.nn.BatchNorm1d(100)
        self.relu1 = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(100, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.pool1(x)

        out1 = self.block1(x)
        out1 = self.pool2(out1)

        out2 = self.block2(out1)

        out3 = self.block3(out2)
        out3 = self.pool3(out3)

        out1 = self.pool3(out1)
        out2 = self.pool3(out2)

        y = torch.cat((out1, out2, out3), 1)

        y = self.squeezer(y)

        y = self.conv2(y)

        y = y.mean(2)
        y = self.lin1(y)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.lin2(y)

        return y
    
class CatLnrV5(nn.Module):
    def __init__(self, n_classes: int=10):
        super(CatLnrV5, self).__init__()

        self.dropout = torch.nn.Dropout(0.5)
        self.conv1 = ConvBnRelu(1, 16, 64, 28, 8, 1)
        self.pool1 = torch.nn.MaxPool1d(2, 2)
        self.block1 = ResBlock(16, 32, 64, 7, False)
        self.pool2 = torch.nn.MaxPool1d(2, 2)
        self.block2 = ResBlock(64, 64, 64, 7, False)
        self.block3 = ResBlock(64, 64, 64, 7, False)
        self.pool3 = torch.nn.MaxPool1d(2, 2)
        self.squeezer = ConvBnRelu(64*3, 16, 1, 0, 1, 1)
        self.conv2 = SeparableConv1D(16, 16, 7)
        self.expander = ConvBnRelu(16, 64*3, 1, 0, 1, 1)

        self.lin1 = torch.nn.Linear(64*3, 100)
        self.bn1 = torch.nn.BatchNorm1d(100)
        self.relu1 = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(100, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.pool1(x)

        out1 = self.block1(x)
        out1 = self.pool2(out1)

        out2 = self.block2(out1)

        out3 = self.block3(out2)
        out3 = self.pool3(out3)

        out1 = self.pool3(out1)
        out2 = self.pool3(out2)

        y = torch.cat((out1, out2, out3), 1)

        y = self.squeezer(y)
        y = self.conv2(y)
        y = self.expander(y)

        y = y.mean(2)
        y = self.lin1(y)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.lin2(y)

        return y

class CatLnrV6(nn.Module):
    def __init__(self, n_classes: int=10):
        super(CatLnrV6, self).__init__()

        self.dropout = torch.nn.Dropout(0.5)
        self.conv1 = ConvBnRelu(1, 16, 64, 28, 8, 1)
        self.pool1 = torch.nn.MaxPool1d(2, 2)
        self.block1 = ResBlock(16, 32, 64, 7, False, False)
        self.pool2 = torch.nn.MaxPool1d(2, 2)
        self.block2 = ResBlock(64, 64, 64, 7, False, False)
        self.block3 = ResBlock(64, 64, 64, 7, False, False)
        self.pool3 = torch.nn.MaxPool1d(2, 2)
        self.conv2 = ConvBnRelu(64, 64, 3, 0, 1, 1)

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
    
class DenseLnr(nn.Module):
    def __init__(self, n_classes: int=10):
        super(DenseLnr, self).__init__()

        growth_rate = 8

        self.dropout = torch.nn.Dropout(0.5)
        self.conv1 = ConvBnRelu(1, 16, 64, 28, 8, 1)
        self.pool1 = torch.nn.MaxPool1d(2, 2)
        self.dense1 = Dense1D(16 + 0 * growth_rate, growth_rate, 7)
        self.dense2 = Dense1D(16 + 1 * growth_rate, growth_rate, 7)
        self.dense3 = Dense1D(16 + 2 * growth_rate, growth_rate, 7)
        self.dense4 = Dense1D(16 + 3 * growth_rate, growth_rate, 7)

        expansion = (16 + 4 * growth_rate)
        squeeze = int(expansion / 2)

        self.conv2 = ConvBnRelu(expansion, squeeze, 3, 1, 1, 1)
        self.pool2 = torch.nn.MaxPool1d(2, 2)

        self.dense5 = Dense1D(squeeze + 0 * growth_rate, growth_rate, 7)
        self.dense6 = Dense1D(squeeze + 1 * growth_rate, growth_rate, 7)
        self.dense7 = Dense1D(squeeze + 2 * growth_rate, growth_rate, 7)
        self.dense8 = Dense1D(squeeze + 3 * growth_rate, growth_rate, 7)

        expansion = (squeeze + 4 * growth_rate)
        squeeze = int(expansion / 2)

        self.conv3 = ConvBnRelu(expansion, squeeze, 3, 1, 1, 1)
        self.pool3 = torch.nn.MaxPool1d(2, 2)

        self.dense9 = Dense1D(squeeze + 0 * growth_rate, growth_rate, 7)
        self.dense10 = Dense1D(squeeze + 1 * growth_rate, growth_rate, 7)
        self.dense11 = Dense1D(squeeze + 2 * growth_rate, growth_rate, 7)
        self.dense12 = Dense1D(squeeze + 3 * growth_rate, growth_rate, 7)
        
        expansion = (squeeze + 4 * growth_rate)
        squeeze = 64

        self.conv4 = ConvBnRelu(expansion, squeeze, 3, 1, 1, 1)

        self.lin1 = torch.nn.Linear(squeeze, 100)
        self.bn1 = torch.nn.BatchNorm1d(100)
        self.relu1 = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(100, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.pool1(x)

        out1 = self.dense1(x)
        out2 = self.dense2(torch.cat((x, out1), 1))
        out3 = self.dense3(torch.cat((x, out1, out2), 1))
        out4 = self.dense4(torch.cat((x, out1, out2, out3), 1))

        x = self.conv2(torch.cat((x, out1, out2, out3, out4), 1))
        x = self.pool2(x)

        out1 = self.dense5(x)
        out2 = self.dense6(torch.cat((x, out1), 1))
        out3 = self.dense7(torch.cat((x, out1, out2), 1))
        out4 = self.dense8(torch.cat((x, out1, out2, out3), 1))

        x = self.conv3(torch.cat((x, out1, out2, out3, out4), 1))
        x = self.pool3(x)

        out1 = self.dense9(x)
        out2 = self.dense10(torch.cat((x, out1), 1))
        out3 = self.dense11(torch.cat((x, out1, out2), 1))
        out4 = self.dense12(torch.cat((x, out1, out2, out3), 1))

        y = torch.cat((x, out1, out2, out3, out4), 1)

        y = self.conv4(y)
        y = y.mean(2)

        y = self.lin1(y)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.lin2(y)

        return y

class LNRNetV2(nn.Module):
    def __init__(self, n_classes: int=10):
        super(LNRNetV2, self).__init__()

        self.dropout = torch.nn.Dropout(0.5)
        self.conv1 = ConvBnRelu(1, 16, 64, 28, 8, 1)
        self.pool1 = torch.nn.MaxPool1d(2, 2)
        self.block1 = ResBlock(16, 32, 64, 7, False)
        self.pool2 = torch.nn.MaxPool1d(2, 2)
        self.block2 = ResBlock(64, 96, 128, 7, False)
        # self.block3 = ResBlock(64, 64, 64, 7, False)
        self.pool3 = torch.nn.MaxPool1d(2, 2)
        # self.block4 = ResBlock(64, 64, 64, 7, False)
        self.block5 = ResBlock(128, 160, 192, 7, False)

        self.lin1 = torch.nn.Linear(192, 100)
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

        # x = self.block3(x)
        x = self.pool3(x)

        # x = self.block4(x)
        x = self.block5(x)
        
        x = x.mean(2)
        x = self.lin1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.lin2(x)

        return x