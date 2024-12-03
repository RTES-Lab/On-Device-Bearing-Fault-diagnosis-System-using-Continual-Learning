import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from typing import Union, Tuple, List

class Conv1dDropout(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: Union[str, Tuple[int, ...]],
        device=None,
        dtype=None,
    ) -> None:
        super(Conv1dDropout, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self._factory_kwargs = {"device": device, "dtype": dtype}
        self._mask = torch.zeros(
            (out_channels, in_channels, kernel_size), **self._factory_kwargs
        )

    def forward(self, input: torch.Tensor, p: float, training: bool) -> torch.Tensor:
        if p < 0 or p > 1:
            raise ValueError("p must be 0~1")
        if training:
            self._mask = self._mask * 0
            random_index = torch.rand((self.kernel_size), **self._factory_kwargs) > p
            self._mask[:, :, random_index] = 1 * (1 / (1 - p + 1e-9))
            masked_weight = self.weight * self._mask
            return F.conv1d(
                input,
                masked_weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        else:
            return F.conv1d(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

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

        if self.residual_connection:
            y = y + self.skip_conv(x)

        return y


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, activation, depthwise, residual_connection):
        super(BottleneckBlock, self).__init__()
        self.activation = activation
        self.residual_connection = residual_connection

        self.layer1 = ConvBnActivation(in_channels, mid_channels, 1, activation, False)
        self.layer2 = ConvBnActivation(mid_channels, mid_channels, kernel_size, activation, depthwise)
        self.layer3 = ConvBnActivation(mid_channels, out_channels, 1, activation, False)

        if residual_connection:
            self.skip_conv = torch.nn.Conv1d(in_channels, out_channels, 1, stride=1, padding=0)
        
    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)

        if self.residual_connection:
            y = y + self.skip_conv(x)

        return y

class MyModelDeep(nn.Module):
    def __init__(self, n_classes: int=10, device: str=None, kernel_size=3, activation=nn.ReLU(), depthwise=False, residual_connection=False, gap=False, first_layer="nodrop"):
        super(MyModelDeep, self).__init__()

        self.dropout_rate = 0.5
        self.cdrop = False

        if first_layer == "cdrop":
            self.first_conv = Conv1dDropout(1, 16, 64, stride=8, padding=28, device=device)
            self.cdrop = True
        elif first_layer == "drop":
            self.first_conv = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Conv1d(1, 16, 64, stride=8, padding=28)
            )
        else:
            self.first_conv = torch.nn.Conv1d(1, 16, 64, stride=8, padding=28)

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

        with torch.no_grad():
            dummy = torch.rand(1, 1, 2048).to(device)
            if self.cdrop:
                dummy = self.first_conv.to(device)(dummy, self.dropout_rate, self.training)
            else:
                dummy = self.first_conv.to(device)(dummy)
            dummy = self.conv_layers.to(device)(dummy)
            dummy = torch.flatten(dummy, 1)
            lin_input = dummy.shape[1]

        if gap:
            self.linear_layers = nn.Sequential(
                torch.nn.AdaptiveAvgPool1d((1)),
                torch.nn.Flatten(),
                torch.nn.Linear(64, 100),
                torch.nn.BatchNorm1d(100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, n_classes),
            )
        else:
            self.linear_layers = nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(lin_input, 100),
                torch.nn.BatchNorm1d(100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, n_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cdrop:
            x = self.first_conv(x, self.dropout_rate, self.training)
        else:
            x = self.first_conv(x)
        x = self.conv_layers(x)
        x = self.linear_layers(x)

        return x

class MyModel(nn.Module):
    def __init__(self, n_classes: int=10, device: str=None, kernel_size=3, activation=nn.ReLU(), depthwise=False, residual_connection=False, gap=False, first_layer="nodrop"):
        super(MyModel, self).__init__()

        self.dropout_rate = 0.5
        self.cdrop = False

        if first_layer == "cdrop":
            self.first_conv = Conv1dDropout(1, 16, 64, stride=8, padding=28, device=device)
            self.cdrop = True
        elif first_layer == "drop":
            self.first_conv = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Conv1d(1, 16, 64, stride=8, padding=28)
            )
        else:
            self.first_conv = torch.nn.Conv1d(1, 16, 64, stride=8, padding=28)

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
            torch.nn.MaxPool1d(2, 2),
            torch.nn.Conv1d(64, 64, 3, stride=1, padding=0),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, 2),
        )

        with torch.no_grad():
            dummy = torch.rand(1, 1, 2048).to(device)
            if self.cdrop:
                dummy = self.first_conv.to(device)(dummy, self.dropout_rate, self.training)
            else:
                dummy = self.first_conv.to(device)(dummy)
            dummy = self.conv_layers.to(device)(dummy)
            dummy = torch.flatten(dummy, 1)
            lin_input = dummy.shape[1]

        if gap:
            self.linear_layers = nn.Sequential(
                torch.nn.AdaptiveAvgPool1d((1)),
                torch.nn.Flatten(),
                torch.nn.Linear(64, 100),
                torch.nn.BatchNorm1d(100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, n_classes),
            )
        else:
            self.linear_layers = nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(lin_input, 100),
                torch.nn.BatchNorm1d(100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, n_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cdrop:
            x = self.first_conv(x, self.dropout_rate, self.training)
        else:
            x = self.first_conv(x)
        x = self.conv_layers(x)
        x = self.linear_layers(x)

        return x

class MyModel2(nn.Module):
    def __init__(self, n_classes: int=10, device: str=None, kernel_size=3, activation=nn.ReLU(),
                 depthwise=False, residual_connection=False, gap=False, first_layer="nodrop",
                 drop_rate=0.5):
        super(MyModel2, self).__init__()

        self.dropout_rate = 0.5
        self.cdrop = False

        if first_layer == "cdrop":
            self.first_conv = Conv1dDropout(1, 16, 64, stride=8, padding=28, device=device)
            self.cdrop = True
        elif first_layer == "drop":
            self.first_conv = torch.nn.Sequential(
                torch.nn.Dropout(drop_rate),
                torch.nn.Conv1d(1, 16, 64, stride=8, padding=28)
            )
        else:
            self.first_conv = torch.nn.Conv1d(1, 16, 64, stride=8, padding=28)

        self.conv_layers = nn.Sequential(
            # conv1
            torch.nn.BatchNorm1d(16),
            activation,
            torch.nn.MaxPool1d(2, 2),
            # conv2
            ResBlock(16, 32, 64, kernel_size, activation, depthwise, residual_connection),
            torch.nn.MaxPool1d(2, 2),
            # conv3
            # ResBlock(64, 64, 64, kernel_size, activation, depthwise, residual_connection),
            # ResBlock(64, 64, 64, kernel_size, activation, depthwise, residual_connection),
            ResBlock(64, 64, 64, kernel_size, activation, depthwise, residual_connection),
            torch.nn.MaxPool1d(2, 2),
            torch.nn.Conv1d(64, 64, kernel_size, stride=1, padding=0),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, 2),
        )

        with torch.no_grad():
            dummy = torch.rand(1, 1, 2048).to(device)
            if self.cdrop:
                dummy = self.first_conv.to(device)(dummy, self.dropout_rate, self.training)
            else:
                dummy = self.first_conv.to(device)(dummy)
            dummy = self.conv_layers.to(device)(dummy)
            dummy = torch.flatten(dummy, 1)
            lin_input = dummy.shape[1]

        if gap:
            self.linear_layers = nn.Sequential(
                torch.nn.AdaptiveAvgPool1d((1)),
                torch.nn.Flatten(),
                torch.nn.Linear(64, 100),
                torch.nn.BatchNorm1d(100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, n_classes),
            )
        else:
            self.linear_layers = nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(lin_input, 100),
                torch.nn.BatchNorm1d(100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, n_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cdrop:
            x = self.first_conv(x, self.dropout_rate, self.training)
        else:
            x = self.first_conv(x)
        x = self.conv_layers(x)
        x = self.linear_layers(x)

        return x


class MyBottleNeckModel(nn.Module):
    def __init__(self, n_classes: int=10, device: str=None, kernel_size=3, activation=nn.ReLU(), depthwise=False, residual_connection=False, gap=False, first_layer="nodrop"):
        super(MyBottleNeckModel, self).__init__()

        self.dropout_rate = 0.5
        self.cdrop = False

        if first_layer == "cdrop":
            self.first_conv = Conv1dDropout(1, 16, 64, stride=8, padding=28, device=device)
            self.cdrop = True
        elif first_layer == "drop":
            self.first_conv = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Conv1d(1, 16, 64, stride=8, padding=28)
            )
        else:
            self.first_conv = torch.nn.Conv1d(1, 16, 64, stride=8, padding=28)

        self.conv_layers = nn.Sequential(
            # conv1
            torch.nn.BatchNorm1d(16),
            activation,
            torch.nn.MaxPool1d(2, 2),
            # conv2
            ResBlock(16, 32, 64, kernel_size, activation, depthwise, residual_connection),
            torch.nn.MaxPool1d(2, 2),
            # conv3
            BottleneckBlock(64, 16, 64, kernel_size, activation, depthwise, residual_connection),
            BottleneckBlock(64, 16, 64, kernel_size, activation, depthwise, residual_connection),
            torch.nn.MaxPool1d(2, 2),
            BottleneckBlock(64, 16, 64, kernel_size, activation, depthwise, residual_connection),
            BottleneckBlock(64, 16, 64, kernel_size, activation, depthwise, residual_connection),
            torch.nn.MaxPool1d(2, 2),
            BottleneckBlock(64, 16, 64, kernel_size, activation, depthwise, residual_connection),
            BottleneckBlock(64, 16, 64, kernel_size, activation, depthwise, residual_connection),
            torch.nn.MaxPool1d(2, 2),
            torch.nn.Conv1d(64, 64, 3, stride=1, padding=0),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, 2),
        )

        with torch.no_grad():
            dummy = torch.rand(1, 1, 2048).to(device)
            if self.cdrop:
                dummy = self.first_conv.to(device)(dummy, self.dropout_rate, self.training)
            else:
                dummy = self.first_conv.to(device)(dummy)
            dummy = self.conv_layers.to(device)(dummy)
            dummy = torch.flatten(dummy, 1)
            lin_input = dummy.shape[1]

        if gap:
            self.linear_layers = nn.Sequential(
                torch.nn.AdaptiveAvgPool1d((1)),
                torch.nn.Flatten(),
                torch.nn.Linear(64, 100),
                torch.nn.BatchNorm1d(100),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(100, n_classes),
            )
        else:
            self.linear_layers = nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(lin_input, 100),
                torch.nn.BatchNorm1d(100),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(100, n_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cdrop:
            x = self.first_conv(x, self.dropout_rate, self.training)
        else:
            x = self.first_conv(x)
        x = self.conv_layers(x)
        x = self.linear_layers(x)

        return x
