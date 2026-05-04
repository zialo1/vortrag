import torch
import torch.nn as nn
import torch.nn.functional as F
'''
version 0.1 monday evening:
takes 64x64 images and upscales them to 128x128.
the training data is half from CELEBA-64 and half
from rf output

'''

# ============================================================
# 1. Time embedding (DDPM: sinusoidal timestep embedding)
# Ho et al. 2020, Section 3
# ============================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        device = t.device

        # sinusoidal embedding as in DDPM paper
        freqs = torch.exp(
            torch.arange(half, device=device) *
            -torch.log(torch.tensor(10000.0)) / half
        )

        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

        return self.mlp(emb)


# ============================================================
# 2. Basic convolutional block
# (used instead of ResNet blocks for minimal DDPM++)
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 3. Downsampling block (U-Net encoder)
# ============================================================

class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = ConvBlock(in_c, out_c)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.block(x)
        return self.pool(x), x


# ============================================================
# 4. Upsampling block (U-Net decoder)
# ============================================================

class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = ConvBlock(in_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


# ============================================================
# 5. Conditional DDPM U-Net for Super-Resolution
#
# This implements:
# p_theta(x_{t-1} | x_t, x_lr)
#
# As in Ho et al. 2020, but extended with conditioning input.
# ============================================================

class SR_DDPM_UNet(nn.Module):
    def __init__(self, base=64, time_dim=128):
        super().__init__()

        # timestep embedding (DDPM paper core idea)
        self.time = TimeEmbedding(time_dim)

        # ----------------------------------------------------
        # CONDITIONING PATH (low-res image x_64 → 128x128)
        # ----------------------------------------------------
        self.cond_net = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.SiLU(),
        )

        # ----------------------------------------------------
        # ENCODER (DDPM U-Net backbone)
        # input: [x_t, conditioning]
        # ----------------------------------------------------
        self.down1 = Down(base * 2, base)
        self.down2 = Down(base, base * 2)

        # middle block
        self.mid = ConvBlock(base * 2, base * 2)

        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------
        self.up1 = Up(base * 4, base)
        self.up2 = Up(base * 2, base)

        self.out = nn.Conv2d(base, 3, 1)

    # ========================================================
    # Forward pass
    # ========================================================
    def forward(self, x_t, x_lr, t):
        """
        x_t  : noisy 128x128 image (DDPM forward process)
        x_lr : 64x64 conditioning image (CelebA or Rectified Flow)
        t    : diffusion timestep
        """

        # ----------------------------------------------------
        # (1) Upsample conditioning to match HR resolution
        # ----------------------------------------------------
        x_lr = F.interpolate(x_lr, size=(128, 128), mode="bilinear")

        cond = self.cond_net(x_lr)

        # ----------------------------------------------------
        # (2) Concatenate noisy image + condition
        # DDPM paper: model learns noise epsilon_theta(x_t, t)
        # ----------------------------------------------------
        x = torch.cat([x_t, cond], dim=1)

        # timestep embedding
        t_emb = self.time(t)

        # ----------------------------------------------------
        # (3) U-Net encoder
        # ----------------------------------------------------
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)

        # inject time embedding (DDPM conditioning on timestep)
        x_mid = self.mid(x2 + t_emb[:, :, None, None])

        # ----------------------------------------------------
        # (4) U-Net decoder
        # ----------------------------------------------------
        x = self.up1(x_mid, skip2)
        x = self.up2(x, skip1)

        # ----------------------------------------------------
        # (5) Predict noise (DDPM objective)
        # Ho et al. 2020: epsilon-prediction
        # ----------------------------------------------------
        return self.out(x)


# ============================================================
# 6. Forward diffusion process (simplified DDPM)
# Ho et al. 2020 Eq. (4)
# ============================================================

def q_sample(x0, t, noise):
    """
    Simplified linear noise schedule (for clarity).
    In DDPM paper, this corresponds to:
    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
    """
    return (1 - t) * x0 + t * noise


# ============================================================
# 7. Training step
# ============================================================

def train_step(model, optimizer, x_lr, x_hr):
    """
    x_lr : 64x64 (CelebA or Rectified Flow output)
    x_hr : 128x128 ground truth
    """

    B = x_hr.size(0)
    device = x_hr.device

    # sample timestep
    t = torch.rand(B, device=device)

    # sample noise
    noise = torch.randn_like(x_hr)

    # forward diffusion (DDPM noising process)
    x_t = q_sample(x_hr, t[:, None, None, None], noise)

    # predict noise (DDPM objective)
    pred = model(x_t, x_lr, t)

    loss = F.mse_loss(pred, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
