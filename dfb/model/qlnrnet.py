import torch

class ConvBnRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups):
        super(ConvBnRelu, self).__init__()
        padding = int((kernel_size-1)/(2))

        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class QLNRNet(torch.nn.Module):
    def __init__(self, n_classes=10):
        super(QLNRNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.dropout = torch.nn.Dropout(0.5)
        self.conv1 = torch.nn.Conv1d(1, 16, 64, 8, 28)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.relu1 = torch.nn.ReLU()

        self.pool1 = torch.nn.MaxPool1d(2, 2)

        self.dconv1 = ConvBnRelu(16, 16, 7, 1, 16)
        self.pconv1 = ConvBnRelu(16, 32, 1, 1, 1)

        self.dconv2 = ConvBnRelu(32, 32, 7, 1, 32)
        self.pconv2 = ConvBnRelu(32, 64, 1, 1, 1)

        self.pool2 = torch.nn.MaxPool1d(2, 2)

        self.dconv3 = ConvBnRelu(64, 64, 7, 1, 64)
        self.pconv3 = ConvBnRelu(64, 64, 1, 1, 1)

        self.dconv4 = ConvBnRelu(64, 64, 7, 1, 64)
        self.pconv4 = ConvBnRelu(64, 64, 1, 1, 1)

        self.dconv5 = ConvBnRelu(64, 64, 7, 1, 64)
        self.pconv5 = ConvBnRelu(64, 64, 1, 1, 1)

        self.dconv6 = ConvBnRelu(64, 64, 7, 1, 64)
        self.pconv6 = ConvBnRelu(64, 64, 1, 1, 1)

        self.pool3 = torch.nn.MaxPool1d(2, 2)

        self.conv2 = torch.nn.Conv1d(64, 64, 3, 1)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.relu2 = torch.nn.ReLU()

        self.flatten = torch.nn.Flatten()
        self.lin1 = torch.nn.Linear(64, 100)
        self.bn3 = torch.nn.BatchNorm1d(100)
        self.relu3 = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(100, n_classes)
    
    def forward(self, x):
        x = self.quant(x)
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.pool1(x)

        x = self.dconv1(x)
        x = self.pconv1(x)

        x = self.dconv2(x)
        x = self.pconv2(x)

        x = self.pool2(x)

        x = self.dconv3(x)
        x = self.pconv3(x)

        x = self.dconv4(x)
        x = self.pconv4(x)

        x = self.dconv5(x)
        x = self.pconv5(x)

        x = self.dconv6(x)
        x = self.pconv6(x)

        x = self.pool3(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = x.mean(2)
        x = self.flatten(x)
        x = self.lin1(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.lin2(x)

        self.dequant(x)

        return x
    
    def fuse_model(self):
        torch.quantization.fuse_modules(self, ['conv1', 'bn1', 'relu1'], inplace=True)
        torch.quantization.fuse_modules(self, ['dconv1.conv', 'dconv1.bn', 'dconv1.relu'], inplace=True)
        torch.quantization.fuse_modules(self, ['pconv1.conv', 'pconv1.bn', 'pconv1.relu'], inplace=True)

        torch.quantization.fuse_modules(self, ['dconv2.conv', 'dconv2.bn', 'dconv2.relu'], inplace=True)
        torch.quantization.fuse_modules(self, ['pconv2.conv', 'pconv2.bn', 'pconv2.relu'], inplace=True)

        torch.quantization.fuse_modules(self, ['dconv3.conv', 'dconv3.bn', 'dconv3.relu'], inplace=True)
        torch.quantization.fuse_modules(self, ['pconv3.conv', 'pconv3.bn', 'pconv3.relu'], inplace=True)

        torch.quantization.fuse_modules(self, ['dconv4.conv', 'dconv4.bn', 'dconv4.relu'], inplace=True)
        torch.quantization.fuse_modules(self, ['pconv4.conv', 'pconv4.bn', 'pconv4.relu'], inplace=True)

        torch.quantization.fuse_modules(self, ['dconv5.conv', 'dconv5.bn', 'dconv5.relu'], inplace=True)
        torch.quantization.fuse_modules(self, ['pconv5.conv', 'pconv5.bn', 'pconv5.relu'], inplace=True)

        torch.quantization.fuse_modules(self, ['dconv6.conv', 'dconv6.bn', 'dconv6.relu'], inplace=True)
        torch.quantization.fuse_modules(self, ['pconv6.conv', 'pconv6.bn', 'pconv6.relu'], inplace=True)
        torch.quantization.fuse_modules(self, ['conv2', 'bn2', 'relu2'], inplace=True)
        torch.quantization.fuse_modules(self, ['lin1', 'bn3'], inplace=True)