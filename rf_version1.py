# -*- coding: utf-8 -*-
"""
v0.1 sonntag nachmittag baseline

"""

# =========================================
# Rectified Flow + DDPM++ U-Net on CelebA-64
# =========================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from PIL import Image


    
device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================
# Time Embedding
# =========================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# =========================================
# Residual Block
# =========================================
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time = nn.Linear(time_dim, out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(t)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


# =========================================
# Attention
# =========================================
class Attention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        h_in = self.norm(x)

        q, k, v = self.qkv(h_in).chunk(3, dim=1)

        q = q.reshape(b, c, -1)
        k = k.reshape(b, c, -1)
        v = v.reshape(b, c, -1)

        attn = torch.softmax((q.transpose(1, 2) @ k) / math.sqrt(c), dim=-1)
        out = (attn @ v.transpose(1, 2)).transpose(1, 2)

        out = out.reshape(b, c, h, w)
        return x + self.proj(out)

# -----------------------------
# 1. Simple UNet (minimal)
# -----------------------------

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, base=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, base),
            nn.ReLU(),
            nn.Linear(base, base)
        )

        self.conv1 = nn.Conv2d(in_channels, base, 3, padding=1)
        self.conv2 = nn.Conv2d(base, base, 3, padding=1)
        self.conv3 = nn.Conv2d(base, in_channels, 3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x, t):
        # t: (B,) → (B,1)
        t = t.view(-1, 1)
        t_embed = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.conv1(x))
        h = h + t_embed
        h = self.act(self.conv2(h))
        out = self.conv3(h)

        return out
    
# =========================================
# DDPM++ U-Net
# =========================================
class UNetDDPM(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()

        time_dim = base * 4

        self.time_embed = nn.Sequential(
            TimeEmbedding(base),
            nn.Linear(base, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init = nn.Conv2d(in_ch, base, 3, padding=1)

        # Down
        self.down1 = ResBlock(base, base * 2, time_dim)      # 64 → 128
        self.down2 = ResBlock(base * 2, base * 4, time_dim)  # 128 → 256

        # Middle
        self.mid = ResBlock(base * 4, base * 4, time_dim)    # 256 → 256
        self.attn_mid = Attention(base * 4)

        # Up (FIXED CHANNELS)
        self.up1 = ResBlock(base * 8, base * 2, time_dim)    # (256+256)=512 → 128
        self.up2 = ResBlock(base * 4, base, time_dim)        # (128+128)=256 → 64

        self.out = nn.Conv2d(base, in_ch, 1)

        self.downsample = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x, t):
        t = self.time_embed(t)

        x = self.init(x)          # 64

        x1 = self.down1(x, t)     # 128
        x1d = self.downsample(x1)

        x2 = self.down2(x1d, t)   # 256
        x2d = self.downsample(x2)

        h = self.mid(x2d, t)
        h = self.attn_mid(h)

        # Up 1
        h = self.upsample(h)              # 256
        h = torch.cat([h, x2], dim=1)     # 256+256=512
        h = self.up1(h, t)                # → 128

        # Up 2
        h = self.upsample(h)              # 128
        h = torch.cat([h, x1], dim=1)     # 128+128=256
        h = self.up2(h, t)                # → 64

        return self.out(h)

# =========================================
# Rectified Flow
# =========================================
class RectifiedFlow:
    def __init__(self, model):
        self.model = model.to(device)
        self.opt = optim.Adam(model.parameters(), lr=1e-4)

    def train_step(self, x1):
        x1 = x1.to(device)
        x0 = torch.randn_like(x1)

        t = torch.rand(x1.size(0), device=device)

        t_ = t[:, None, None, None]

        xt = (1 - t_) * x0 + t_ * x1
        v_target = x1 - x0

        v_pred = self.model(xt, t)

        loss = ((v_pred - v_target) ** 2).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    @torch.no_grad()
    def sample(self, n=16, steps=50):
        x = torch.randn(n, 3, 64, 64).to(device)
        dt = 1.0 / steps

        for i in range(steps):
            t = torch.full((n,), i / steps, device=device)
            x = x + dt * self.model(x, t)

        return x


# =========================================
# Trainer
# =========================================
class Trainer:
    def __init__(self, rf, loader):
        self.rf = rf
        self.loader = loader

    def train(self, epochs):
        for ep in range(epochs):
            pbar = tqdm(self.loader)

            for x, _ in pbar:
                loss = self.rf.train_step(x)
                pbar.set_description(f"loss {loss:.4f}")

            torch.save(self.rf.model.state_dict(), f"rf_ddpmpp_celeba64_{ep}.pth")
            self.show(ep)

    def show(self, ep):
        imgs = self.rf.sample(16)
        imgs = (imgs.clamp(-1, 1) + 1) / 2

        fig, ax = plt.subplots(4, 4, figsize=(6, 6))
        for i, a in enumerate(ax.flatten()):
            a.imshow(imgs[i].permute(1, 2, 0).cpu())
            a.axis("off")

        plt.suptitle(f"Epoch {ep}")
        plt.show()


# =========================================
# Dataset: CelebA-64
# =========================================
def get_celeba64():
    return CelebA64("../data/celeba")

# unused
def get_online_celeba64():

    transform = transforms.Compose([
        transforms.Resize(64),        # key change
        transforms.CenterCrop(64),    # ensures exact 64x64
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))

    ])

    return datasets.CelebA(
        root="./data",
        split="train",
        download=True,
        transform=transform

    )


class CelebA64(Dataset):
    def __init__(self, root="data/celeba"):
        self.root = root
        self.files = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.endswith((".jpg", ".png", ".jpeg"))
        ]

        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.transform(img)
        return img, 0   # dummy label (unused)
    
    

# =========================================
# Main
# =========================================
def main():
    dataset = get_celeba64()
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    model = SimpleUNet().to(device)
#    model = UNetDDPM().to(device)
    rf = RectifiedFlow(model)

    print("trainer runs..")
    trainer = Trainer(rf, loader)
    trainer.train(epochs=5)


if __name__ == "__main__":
    main()
    


