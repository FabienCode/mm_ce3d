import torch
import torch.nn as nn

from mmdet.models import NECKS

class SharedMLP(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            transpose=False,
            padding_mode='zeros',
            bn=False,
            activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network

            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


@NECKS.register_module()
class PointAttentionNetwork(nn.Module):
    def __init__(self, C, ratio):
        super(PointAttentionNetwork, self).__init__()
        self.bn1 = nn.BatchNorm1d(C // ratio)
        self.bn2 = nn.BatchNorm1d(C // ratio)
        self.bn3 = nn.BatchNorm1d(C)

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C // ratio, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C // ratio, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.ReLU())

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, label, ori):
        b, c, n = ori.shape

        a = self.conv1(label).permute(0, 2, 1)  # b, n, c/ratio

        b = self.conv2(ori)  # b, c/ratio, n

        s = self.softmax(torch.bmm(a, b))  # b,n,n

        d = self.conv3(label)  # b,c,n
        out = label + torch.bmm(d, s.permute(0, 2, 1))

        return out

@NECKS.register_module()
class gtfeature(nn.Module):
    def __init__(self):
        self.SA_modules = nn.ModuleList()
        self.mlp1 = SharedMLP(18, 64, bn=True, activation_fn=nn.ReLU())
        self.mlp2 = SharedMLP(64, 256, bn=True, activation_fn=nn.ReLU())
        self.mlp3 = SharedMLP(256, 256, bn=True, activation_fn=nn.ReLU())