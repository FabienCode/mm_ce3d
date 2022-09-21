import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

layer_feature = []
layer1 = []
layer1.append(torch.randn((8, 64, 2048)))
layer1.append(torch.randn((8, 64, 2048)))
layer1.append(torch.randn((8, 128, 2048)))
layer_feature.append(layer1)
layer2 = []
layer2.append(torch.randn((8, 128, 1024)))
layer2.append(torch.randn((8, 128, 1024)))
layer2.append(torch.randn((8, 256, 1024)))
layer_feature.append(layer2)
layer3 = []
layer3.append(torch.randn((8, 128, 512)))
layer3.append(torch.randn((8, 128, 512)))
layer3.append(torch.randn((8, 256, 512)))
layer_feature.append(layer3)
layer4 = []
layer4.append(torch.randn((8, 128, 256)))
layer4.append(torch.randn((8, 128, 256)))
layer4.append(torch.randn((8, 256, 256)))
layer_feature.append(layer4)

layer_xyz = []
layer_xyz.append(torch.randn((8, 2048, 3)))
layer_xyz.append(torch.randn((8, 1024, 3)))
layer_xyz.append(torch.randn((8, 512, 3)))
layer_xyz.append(torch.randn((8, 256, 3)))

layer_xyz_feature = []
layer_xyz1 = []
layer_xyz1.append(torch.randn((8, 2048, 64)))
layer_xyz1.append(torch.randn((8, 2048, 64)))
layer_xyz1.append(torch.randn((8, 2048, 128)))
layer_xyz_feature.append(layer_xyz1)
layer_xyz2 = []
layer_xyz2.append(torch.randn((8, 1024, 128)))
layer_xyz2.append(torch.randn((8, 1024, 128)))
layer_xyz2.append(torch.randn((8, 1024, 256)))
layer_xyz_feature.append(layer_xyz2)
layer_xyz3 = []
layer_xyz3.append(torch.randn((8, 512, 128)))
layer_xyz3.append(torch.randn((8, 512, 128)))
layer_xyz3.append(torch.randn((8, 512, 256)))
layer_xyz_feature.append(layer_xyz3)
layer_xyz4 = []
layer_xyz4.append(torch.randn((8, 256, 128)))
layer_xyz4.append(torch.randn((8, 256, 128)))
layer_xyz4.append(torch.randn((8, 256, 256)))
layer_xyz_feature.append(layer_xyz4)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_length: int):
    "feature map to windows"
    B, N, C = x.shape
    x = x.view(B, N // window_length, window_length, C)
    windows = x.contiguous().view(-1, window_length, C)
    return windows


def window_reverse(windows, windows_length, N):
    B = int(windows.shape[0] / (N / windows_length))
    x = windows.view(B, N // windows_length, windows_length, -1)
    x = x.contiguous().view(B, N, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.softmax = nn.Softmax(dim=-1)
        # self.embedding = nn.Linear(3, dim)

    def forward(self, q, kv, attn_mask):
        B, N, C = q.shape
        attn = (q @ kv.transpose(-2, -1))
        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(B // nW, nW, N, N) + attn_mask.unsqueeze(0)
            attn = attn.view(-1, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        x = (attn @ kv).transpose(1, 2).reshape(B, N, C)

        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, window_size=32, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q, kv, attn_mask):
        B, N, C = q.shape

        shortcut = kv
        q = self.norm1(q)
        kv = self.norm1(kv)

        # cyclic shift
        if self.shift_size > 0:
            shifted_q = torch.roll(q, shifts=-self.shift_size, dims=1)
            shifted_kv = torch.roll(kv, shifts=-self.shift_size, dims=1)
        else:
            shifted_q = q
            shifted_kv = kv
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_q, self.window_size)
        kv_windows = window_partition(shifted_kv, self.window_size)
        # x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, kv_windows, attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        # attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_kv = window_reverse(attn_windows, self.window_size, N)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            kv = torch.roll(shifted_kv, shifts=self.shift_size, dims=1)
        else:
            kv = shifted_kv

        # FFN
        kv = shortcut + self.drop_path(kv)
        kv = kv + self.drop_path(self.mlp(self.norm2(kv)))
        # kv = kv + self.drop_path(self.mlp(kv))

        return kv


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.
    """

    def __init__(self, dim, depth, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        # self.blocks = nn.ModuleList([
        #     SwinTransformerBlock(
        #         dim=dim,
        #         window_size=window_size,
        #         shift_size=0 if (i % 2 == 0) else self.shift_size,
        #         mlp_ratio=mlp_ratio,
        #         qkv_bias=qkv_bias,
        #         drop=drop,
        #         attn_drop=attn_drop,
        #         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
        #         norm_layer=norm_layer)
        #     for i in range(depth)])
        # self.block = nn.ModuleList()
        # for i in range(depth):
        #     self.block.append(SwinTransformerBlock(dim=dim, window_size=window_size,
        #                                            shift_size=0 if (i % 2 == 0) else self.shift_size))
        # self.block = nn.Sequential(
        #     SwinTransformerBlock(dim=dim, window_size=window_size),
        #     SwinTransformerBlock(dim=dim, window_size=window_size, shift_size=self.shift_size)
        # )
        self.block1 = SwinTransformerBlock(dim=dim, window_size=window_size)
        self.block2 = SwinTransformerBlock(dim=dim, window_size=window_size, shift_size=self.shift_size)
        self.downsample = None
        # self.xyz_embeddings = nn.Linear(3, dim)

    def create_mask(self, q, N):
        # calculate attention mask for SW-MSA
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, N, 1), device=q.device)
        q_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in q_slices:
            img_mask[:, h, :] = cnt
            cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, q, kv, N):
        # q = self.xyz_embeddings(q)
        attn_mask = self.create_mask(q, N)  # [nW, Mh*Mw, Mh*Mw]
        x = self.block1(q, kv, attn_mask)
        x = self.block2(q, x, attn_mask)
        # for blk in self.blocks:
        #     x = blk(q, kv, attn_mask)

        return x


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


if __name__ == '__main__':
    windows = window_partition(layer_feature[0][0].permute(0, 2, 1).contiguous(), 32)
    window_reverse(windows, 32, 2048)
    # swin_block = SwinTransformerBlock(64)
    # a = swin_block(layer_xyz_feature[0][0], layer_feature[0][0].transpose(-2, -1))
    swin_bs = BasicLayer(64, 2, 32)
    a = swin_bs(layer_xyz_feature[0][0], layer_feature[0][0].transpose(-2, -1), 2048)
