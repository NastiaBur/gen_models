import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class ResBlk(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=False, act=nn.LeakyReLU(0.2), use_norm=True):
        super().__init__()
        self.downsample = downsample
        self.act = act

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.norm1 = nn.InstanceNorm2d(out_channels, affine=False) if use_norm else nn.Identity()
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=False) if use_norm else nn.Identity()

        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        self.pool = nn.AvgPool2d(2) if downsample else nn.Identity()

    def forward(self, x):
        # Skip path
        h_skip = x
        if self.skip is not None:
            h_skip = self.skip(h_skip)
        h_skip = self.pool(h_skip)

        # Main path
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.pool(h)

        h = self.conv2(h)
        h = self.norm2(h)

        return h + h_skip


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.num_features = num_features
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)
        # nn.init.zeros_(self.fc.weight)
        # nn.init.zeros_(self.fc.bias)

    def forward(self, x, s):
        h = self.norm(x)  # [B, C, H, W]
        style = self.fc(s)  # [B, 2C]
        gamma, beta = style.chunk(2, dim=1)  # each [B, C]

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return (1.0 + gamma) * h + beta


class AdaINResBlk(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim=64, upsample=False, act=nn.LeakyReLU(0.2)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        self.act = act if act is not None else nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.adain1 = AdaIN(style_dim, out_channels)
        self.adain2 = AdaIN(style_dim, out_channels)

        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def _upsample(self, x):
        if not self.upsample:
            return x
        return F.interpolate(x, scale_factor=2, mode="nearest")

    def forward(self, x, s):
        # Skip
        h_skip = self._upsample(x)
        if self.skip is not None:
            h_skip = self.skip(h_skip)

        # Main
        h = self._upsample(x)
        h = self.conv1(h)
        h = self.adain1(h, s)
        h = self.act(h)

        h = self.conv2(h)
        h = self.adain2(h, s)

        return h + h_skip


class Generator(nn.Module):
    def __init__(self, img_channels=3, dim=64, style_dim=64, max_dim=512):
        super().__init__()
        self.img_channels = img_channels
        self.dim = dim
        self.style_dim = style_dim

        self.from_rgb = nn.Conv2d(img_channels, dim, kernel_size=3, padding=1)

        # -------------------------
        # Encoder: 256 -> 128 -> 64 -> 32 -> 16
        # -------------------------
        enc_blocks = []
        ch = dim
        for _ in range(4):
            ch_next = min(ch * 2, max_dim)
            enc_blocks.append(ResBlk(ch, ch_next, downsample=True))
            ch = ch_next
        self.encoder = nn.ModuleList(enc_blocks)

        self.bottleneck = nn.Sequential(
            ResBlk(ch, ch, downsample=False),
            ResBlk(ch, ch, downsample=False),
        )

        # -------------------------
        # Decoder: 16 -> 32 -> 64 -> 128 -> 256
        # -------------------------
        dec_blocks = []
        for _ in range(4):
            ch_next = max(ch // 2, dim)
            dec_blocks.append(AdaINResBlk(ch, ch_next, style_dim=style_dim, upsample=True))
            ch = ch_next
        self.decoder = nn.ModuleList(dec_blocks)

        # To RGB
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, img_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x, s):
        # x: [B,3,256,256], s: [B,style_dim]
        h = self.from_rgb(x)

        # encode content
        for blk in self.encoder:
            h = blk(h)

        h = self.bottleneck(h)

        # decode with style injection
        for blk in self.decoder:
            h = blk(h, s)

        out = self.to_rgb(h)
        return out


class StyleEncoder(nn.Module):

    def __init__(self, img_channels=3, dim=64, style_dim=64, num_domains=2, max_dim=512, img_size=256):
        super().__init__()
        self.img_channels = img_channels
        self.dim = dim
        self.style_dim = style_dim
        self.num_domains = num_domains

        self.from_rgb = nn.Conv2d(img_channels, dim, kernel_size=3, padding=1)

        # Downsample: for 256 -> 128 -> 64 -> 32 -> 16 -> 8
        blocks = []
        ch = dim
        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            ch_next = min(ch * 2, max_dim)
            blocks.append(ResBlk(ch, ch_next, downsample=True, use_norm=False))
            ch = ch_next
        self.encoder = nn.Sequential(*blocks)

        # A little bottleneck (optional but often helps)
        self.bottleneck = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Domain-specific style heads:
        # output is [B, num_domains, style_dim], then select by y
        self.style_heads = nn.Conv2d(ch, num_domains * style_dim, kernel_size=1, padding=0)

    def forward(self, x_ref, y):

        h = self.from_rgb(x_ref)
        h = self.encoder(h)
        h = self.bottleneck(h)

        # Global average pooling -> [B, C, 1, 1]
        h = h.mean(dim=(2, 3), keepdim=True)

        # Produce domain-specific style vectors: [B, num_domains*style_dim, 1, 1]
        out = self.style_heads(h)

        # Reshape to [B, num_domains, style_dim]
        B = out.size(0)
        out = out.view(B, self.num_domains, self.style_dim)

        # Select the style corresponding to domain y
        s = out[torch.arange(B, device=out.device), y]  # [B, style_dim]
        return s


class MappingNetwork(nn.Module):

    def __init__(self, latent_dim=16, hidden_dim=256, style_dim=64, num_domains=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.style_dim = style_dim
        self.num_domains = num_domains

        layers = []
        in_dim = latent_dim
        for _ in range(4):
            layers += [nn.Linear(in_dim, hidden_dim)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            in_dim = hidden_dim
        self.shared = nn.Sequential(*layers)

        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, style_dim) for _ in range(num_domains)
        ])

    def forward(self, z, y):
        h = self.shared(z)  # [B, hidden_dim]

        # Compute all domain outputs and select by y
        # Stack: [B, num_domains, style_dim]
        out = torch.stack([head(h) for head in self.heads], dim=1)

        # Select the correct domain style for each sample: [B, style_dim]
        B = z.size(0)
        s = out[torch.arange(B, device=out.device), y]
        return s


class Discriminator(nn.Module):

    def __init__(self, img_channels=3, dim=64, num_domains=2, max_dim=512, img_size=256):
        super().__init__()
        self.num_domains = num_domains

        self.from_rgb = nn.Conv2d(img_channels, dim, kernel_size=3, padding=1)

        blocks = []
        ch = dim

        # 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4
        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            ch_next = min(ch * 2, max_dim)
            blocks.append(ResBlk(ch, ch_next, downsample=True, use_norm=False))
            ch = ch_next

        self.blocks = nn.Sequential(*blocks)

        self.final_conv = nn.Conv2d(ch, num_domains, kernel_size=1)

    def forward(self, x, y):
        h = self.from_rgb(x)
        h = self.blocks(h)

        h = self.final_conv(h)   # [B, num_domains, H, W]
        h = h.sum(dim=(2, 3))    # [B, num_domains]

        B = x.size(0)
        out = h[torch.arange(B, device=x.device), y]  # select domain
        return out
