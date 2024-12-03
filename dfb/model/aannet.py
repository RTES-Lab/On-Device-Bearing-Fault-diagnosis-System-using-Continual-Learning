import torch
from torch import nn
import torch.nn.functional as F

class AttentionWeight(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super(AttentionWeight, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hs, ht):
        score = torch.bmm(self.attn(ht).transpose(1, 0),
                          hs.permute(0, 2, 1))
        return F.softmax(score, dim=1)

class Attention(nn.Module):
    def __init__(self, hidden_size: int,
                 output_size: int) -> None:
        super(Attention, self).__init__()
        self.concat_layer = nn.Linear(hidden_size*2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.attn = AttentionWeight(hidden_size)

    def forward(self, hs, ht):
        attn_weight = self.attn(hs, ht)
        context = torch.bmm(attn_weight, hs).squeeze(1)
        
        concat_input = torch.cat((ht.squeeze(0), context), 1)
        concat_output = torch.tanh(self.concat_layer(concat_input))

        return concat_output

class AAnNet(nn.Module):
    def __init__(self, n_classes: int=10):
        super(AAnNet, self).__init__()
        self.first_dropout = nn.Dropout(p=0.5)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=128, stride=1, padding='same'),
            nn.BatchNorm1d(num_features=64),
            nn.ELU()
        )
        self.dropout1 = nn.Dropout(p=0.5)
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=72, kernel_size=128, stride=1, padding='same'),
            nn.BatchNorm1d(num_features=72),
            nn.ELU()
        )
        self.dropout2 = nn.Dropout(p=0.5)
        self.gru1 = nn.GRU(input_size=72, hidden_size=64, batch_first=True)
        self.gru2 = nn.GRU(input_size=64, hidden_size=128, batch_first=True)
        self.dropout3 = nn.Dropout(p=0.5)
        self.attention = Attention(hidden_size=128, output_size=128)

        self.linear_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, x):
        x = self.first_dropout(x)
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = torch.permute(x, (0, 2, 1))
        x, _ = self.gru1(x)
        hs, ht = self.gru2(x)
        hs = self.dropout3(hs)
        ht = self.dropout3(ht)
        out = self.attention(hs, ht)
        out = self.linear_layers(out)

        return out