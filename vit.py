import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from config import Config
from utils import check_tuple


class Attention(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        # base param
        self.dim = config.dim
        self.n_head = config.n_head
        assert self.dim % self.n_head == 0, f'dim must be divisible by n_head'
        self.head_dim = self.dim // self.n_head
        self.scale = self.head_dim ** -0.5
        self.dropout_rate = config.dropout_rate

        # base module
        self.to_qkv = nn.Linear(self.dim, 3 * self.n_head * self.head_dim, bias = False)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.to_out = nn.Linear(self.n_head * self.head_dim, self.dim, bias = False)

    def forward(self, x, mask = None):
        # x: [b, n, d]
        x = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h = self.n_head), x)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim = -1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class FFN(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        # base param
        self.dim = config.dim
        self.dropout_rate = config.dropout_rate

        # base module
        self.net = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.dim * 4, self.dim),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, x):
        # x: [b, n, d]
        return self.net(x)


class Transformer(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        # base param
        self.dim = config.dim
        self.n_layer = config.n_layer

        # base module
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.layer = nn.ModuleList([])
        for _ in range(self.n_layer):
            self.layer.append(
                nn.ModuleList([
                    Attention(config),
                    FFN(config)
                ])
            )

    def forward(self, x):
        # x: [b, n, d]
        for attn, ffn in self.layer:
            x = attn(self.norm1(x)) + x
            x = ffn(self.norm2(x)) + x
        return x


class VIT(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        # base param
        self.img_h, self.img_w = check_tuple(config.img_size) # img size
        self.p_h, self.p_w = check_tuple(config.patch_size) # patch size
        self.channel = config.channel # img channel
        self.dim = config.dim # model dim
        assert self.img_h % self.p_h == 0 and self.img_w % self.p_w == 0, f'img_size must be divisible by patch_size'
        self.num_patch = (self.img_h // self.p_h) * (self.img_w // self.p_w) # num of patch
        self.patch_dim = self.channel * self.p_h * self.p_w # patch dim
        self.dropout_rate = config.dropout_rate # dropout rate
        self.mode = config.mode # pool mode
        assert self.mode in ['cls', 'mean'], f'mode must be cls or mean'
        self.n_class = config.n_class # num of class

        # base module
        self.to_patch = nn.Sequential(
            Rearrange('b c (h p_h) (w p_w) -> b (h w) (c p_h p_w)', p_h = self.p_h, p_w = self.p_w),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, self.dim),
            nn.LayerNorm(self.dim)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patch + 1, self.dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(self.dropout_rate)
        self.transformer = Transformer(config)
        self.to_out = nn.Linear(self.dim, self.n_class)

    def forward(self, x):
        # x: [b, c, h, w]
        x = self.to_patch(x)

        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b = x.shape[0])
        x = torch.cat((cls_token, x), dim = 1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        x = x[:, 0] if self.mode == 'cls' else x.mean(dim = 1)
        out = self.to_out(x)
        return out


if __name__ == '__main__':
    model = VIT(Config())
    x = torch.randn(1, 3, 224, 224)
    model(x)
