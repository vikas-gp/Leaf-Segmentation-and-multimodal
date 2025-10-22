"""
model.py — UNet + Attention gates + GraphReasoner (runnable)

Design summary:
- Spatial AttentionBlock at E1 and E2 (skip gating).
- Compact SelfAttentionGate at E3 (reduced Q/K channels, value conv, residual scalar gamma).
- GraphReasoner placed after dec2 (mid-decoder) to improve connectivity.
- CLAHE created in __getitem__ to avoid DataLoader pickling issues.
- HybridLoss weights: BCE=0.45, FocalTversky=0.20, Topology=0.35.
- Training defaults: BATCH_SIZE=8, SEEDS=[42,7], checkpoint naming includes seed and batch size.
- Model returns (mask_logits, (sk_d1, sk_d2)) for compatibility.
"""

import os
import random
from typing import Tuple, List

import numpy as np
from PIL import Image
import cv2
from skimage.morphology import skeletonize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -------------------------
# Defaults & utility funcs
# -------------------------
BATCH_SIZE = 8
SEEDS = [42, 7]
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_ckpt_name(seed: int, prefix: str = "ckpt_unet_allself_bs"):
    return f"{prefix}{BATCH_SIZE}_s{seed}.pth"


# -------------------------
# Dataset (CLAHE per-getitem)
# -------------------------
class LeafVeinDataset(Dataset):
    """
    Dataset that reads images and masks, optionally applies CLAHE (created inside __getitem__
    to avoid DataLoader multiprocessing pickling issues). Returns:
        image_t: Tensor (3,H,W) float32 normalized by transforms
        mask_t: Tensor (1,H,W) float32 in {0,1}
        sk_t: Tensor (1,H,W) float32 skeleton
    """
    def __init__(self, img_dir: str, mask_dir: str, transform=None, use_clahe: bool = True):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.use_clahe = use_clahe
        self.img_names = sorted([f for f in os.listdir(img_dir) if not f.startswith('.')])

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, idx: int):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        base, _ = os.path.splitext(img_name)

        # find mask with typical extensions
        mask_path = None
        for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'):
            p = os.path.join(self.mask_dir, base + ext)
            if os.path.exists(p):
                mask_path = p
                break
        if mask_path is None:
            raise FileNotFoundError(f"No corresponding mask for {img_name}")

        # Read image (RGB)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create CLAHE per-call to avoid pickling issues in multiprocessing DataLoader
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        pil_img = Image.fromarray(image)
        if self.transform:
            image_t = self.transform(pil_img)
        else:
            image_t = transforms.ToTensor()(pil_img)

        # Mask
        mask_np = np.array(Image.open(mask_path))
        if mask_np.ndim == 3:
            mask_np = mask_np[..., :3].max(axis=2)
        mask_np = (mask_np / 255.0).astype(np.float32)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).float()

        # Skeleton GT: optional precomputed _sk.png else compute
        sk_path = os.path.join(self.mask_dir, base + "_sk.png")
        if os.path.exists(sk_path):
            sk_np = np.array(Image.open(sk_path)).astype(np.float32) / 255.0
        else:
            sk_np = skeletonize(mask_np > 0.5).astype(np.float32)
        sk_t = torch.from_numpy(sk_np).unsqueeze(0).float()

        return image_t, mask_t, sk_t


# -------------------------
# Basic conv block (GroupNorm)
# -------------------------
def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
    )


# -------------------------
# Spatial AttentionBlock (for E1 and E2)
# -------------------------
class AttentionBlock(nn.Module):
    """
    Spatial attention gate used at skip connections (lightweight).
    g: decoder (gate), x: encoder skip
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal (decoder), x: skip (encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# -------------------------
# Compact SelfAttentionGate (for E3)
# -------------------------
class SelfAttentionGate(nn.Module):
    """
    Compact Self-Attention Gate used at E3 skip connection.
    - Reduces Q/K channels by factor 'reduction'
    - Projects value via 1x1 conv
    - Outputs refined skip feature (same channels as x) with residual scalar gamma
    Inputs:
        g: decoder gate feature (B, Cg, H, W)
        x: encoder skip (B, Cx, H, W)
    """
    def __init__(self, in_channels: int, gate_channels: int, reduction: int = 4):
        super().__init__()
        self.reduced_ch = max(1, in_channels // reduction)
        # query from gate g, key from x, value from x
        self.query_conv = nn.Conv2d(gate_channels, self.reduced_ch, kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(in_channels, self.reduced_ch, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, g, x):
        B, Cx, H, W = x.shape
        _, Cg, Hg, Wg = g.shape
        if (Hg, Wg) != (H, W):
            g = F.interpolate(g, size=(H, W), mode='bilinear', align_corners=False)

        q = self.query_conv(g).view(B, self.reduced_ch, H * W)   # B, Cr, N
        k = self.key_conv(x).view(B, self.reduced_ch, H * W)     # B, Cr, N
        v = self.value_conv(x).view(B, Cx, H * W)                # B, Cx, N

        # attention N x N but with reduced channel dims for q/k
        attn = torch.bmm(q.permute(0, 2, 1), k)                  # B, N, N
        attn = self.softmax(attn / (self.reduced_ch ** 0.5))
        out = torch.bmm(v, attn)                                 # B, Cx, N
        out = out.view(B, Cx, H, W)
        out = self.out_proj(out)
        return self.gamma * out + x


# -------------------------
# Graph Reasoner (after dec2)
# -------------------------
class GraphReasoner(nn.Module):
    """
    Lightweight graph reasoning on pooled spatial grid to enforce connectivity.
    Uses adaptive pooling to pool to a small graph size, does multi-head message passing,
    then upsamples and projects back.
    """
    def __init__(self, in_channels: int, pool_size: int = 16, reduction: int = 2, heads: int = 2, iters: int = 1):
        super().__init__()
        self.pool_size = pool_size
        self.heads = heads
        self.iters = iters
        hidden = max(8, in_channels // reduction)

        self.q_proj = nn.Conv2d(in_channels, hidden * heads, 1, bias=False)
        self.k_proj = nn.Conv2d(in_channels, hidden * heads, 1, bias=False)
        self.v_proj = nn.Conv2d(in_channels, hidden * heads, 1, bias=False)
        self.out_proj = nn.Conv2d(hidden * heads, in_channels, 1, bias=False)

        self.norm = nn.GroupNorm(8, in_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q_pool = F.adaptive_avg_pool2d(q, (self.pool_size, self.pool_size))
        k_pool = F.adaptive_avg_pool2d(k, (self.pool_size, self.pool_size))
        v_pool = F.adaptive_avg_pool2d(v, (self.pool_size, self.pool_size))

        S = self.pool_size * self.pool_size
        # reshape: B, heads, hidden, S
        q_pool = q_pool.view(B, self.heads, -1, S)
        k_pool = k_pool.view(B, self.heads, -1, S)
        v_pool = v_pool.view(B, self.heads, -1, S)

        msg = v_pool
        for _ in range(self.iters):
            attn = torch.einsum("bhcs,bhct->bhst", q_pool, k_pool) / (q_pool.size(2) ** 0.5)
            attn = F.softmax(attn, dim=-1)
            msg = torch.einsum("bhst,bhct->bhcs", attn, v_pool)
            v_pool = msg

        out_pool = msg.reshape(B, -1, S).view(B, -1, self.pool_size, self.pool_size)
        out_up = F.interpolate(out_pool, size=(H, W), mode='bilinear', align_corners=False)
        out = self.out_proj(out_up)
        return self.act(self.norm(out + x))


# -------------------------
# UNet: E1/E2 spatial gates, E3 self-attention gate, GraphReasoner at dec2
# -------------------------
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = conv_block(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        # Attention gates:
        # spatial gates (E1, E2) are lighter and operate via AttentionBlock
        # self-attention gate (E3) is compact (reduced q/k) to limit memory
        self.att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)  # for e2 (g: d2, x: e2)
        self.att1 = AttentionBlock(F_g=32, F_l=32, F_int=16)  # for e1 (g: d1, x: e1)
        self.sa3_gate = SelfAttentionGate(in_channels=128, gate_channels=128, reduction=4)

        # Bottleneck
        self.bottleneck = conv_block(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)

        # GraphReasoner after dec2
        self.graph_reasoner_d2 = GraphReasoner(64, pool_size=16, heads=2, reduction=2, iters=1)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)

        # Heads
        self.final = nn.Conv2d(32, 1, 1)
        self.sk_head_d2 = nn.Conv2d(64, 1, 1)
        self.sk_head_d1 = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Encoder
        e1 = self.enc1(x)                 # B,32,H,W
        e2 = self.enc2(self.pool1(e1))    # B,64,H/2,W/2
        e3 = self.enc3(self.pool2(e2))    # B,128,H/4,W/4

        b = self.bottleneck(self.pool3(e3))  # B,256,H/8,W/8

        # Decoder stage 3
        d3 = self.up3(b)                       # B,128,H/4,W/4
        # refine encoder skip e3 using self-attention gate (compact)
        e3_ref = self.sa3_gate(d3, e3)         # B,128,H/4,W/4
        d3 = self.dec3(torch.cat([d3, e3_ref], dim=1))

        # Decoder stage 2
        d2 = self.up2(d3)                      # B,64,H/2,W/2
        # spatial attention gate at e2 (lightweight)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        # GraphReasoner refinement (mid-decoder)
        d2 = self.graph_reasoner_d2(d2)

        # Decoder stage 1
        d1 = self.up1(d2)                      # B,32,H,W
        # spatial attention gate at e1 (lightweight)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        mask_logits = self.final(d1)  # B,1,H,W
        sk_d1 = self.sk_head_d1(d1)
        sk_d2 = F.interpolate(self.sk_head_d2(d2), size=mask_logits.shape[2:], mode='bilinear', align_corners=False)

        return mask_logits, (sk_d1, sk_d2)


# -------------------------
# Losses
# -------------------------
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.33, smooth: float = 1e-6):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        tp = (probs * targets).sum()
        fp = ((1 - targets) * probs).sum()
        fn = (targets * (1 - probs)).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky) ** self.gamma


class TopologyLoss(nn.Module):
    """
    Topology loss using Laplacian / edge consistency with an epoch-scaled weight.
    """
    def __init__(self, start_w: float = 1.0, end_w: float = 8.0, max_epoch: int = 100):
        super().__init__()
        self.start_w = start_w
        self.end_w = end_w
        self.max_epoch = max_epoch
        self.register_buffer('laplace', torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, epoch: int = 1) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        laplace = self.laplace.to(dtype=probs.dtype, device=probs.device)  # convert dtype/device here
        edge_pred = torch.abs(F.conv2d(probs, laplace, padding=1))
        edge_gt = torch.abs(F.conv2d(targets, laplace, padding=1))
        w = min(self.end_w, self.start_w + (self.end_w - self.start_w) * (epoch / max(1, self.max_epoch)))
        return w * torch.mean((edge_pred - edge_gt) ** 2)


def edge_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    laplace = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=pred.device, dtype=pred.dtype).unsqueeze(0).unsqueeze(0)
    edge_p = F.conv2d(torch.sigmoid(pred), laplace, padding=1)
    edge_t = F.conv2d(target, laplace, padding=1)
    return torch.mean((edge_p - edge_t) ** 2)


def soft_junction_loss(sk_logits: torch.Tensor, sk_gt: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(sk_logits)
    kernel = torch.tensor([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]], device=probs.device, dtype=probs.dtype).unsqueeze(0).unsqueeze(0)
    sk_gt = sk_gt.float()
    neigh = F.conv2d(probs, kernel, padding=1)
    junc_pred = torch.sigmoid((neigh - 2.5) * 5.0)
    neigh_gt = F.conv2d(sk_gt, kernel, padding=1)
    junc_gt = (neigh_gt >= 3).float()
    return F.binary_cross_entropy(junc_pred, junc_gt)



class HybridLoss(nn.Module):
    """
    HybridLoss: BCEWithLogits (0.45) + FocalTversky (0.20) + Topology (0.35) by default.
    """
    def __init__(self, bce_w: float = 0.45, ft_w: float = 0.20, topo_w: float = 0.35):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.ftl = FocalTverskyLoss()
        self.topo = TopologyLoss(start_w=0.5, end_w=8.0, max_epoch=200)
        self.bce_w = bce_w
        self.ft_w = ft_w
        self.topo_w = topo_w

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, epoch: int = 1) -> torch.Tensor:
        bce_term = self.bce(logits, targets)
        ft_term = self.ftl(logits, targets)
        topo_term = self.topo(logits, targets, epoch)
        return self.bce_w * bce_term + self.ft_w * ft_term + self.topo_w * topo_term


# -------------------------
# Training loop
# -------------------------
def train_one_seed(seed: int,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   device: torch.device,
                   num_epochs: int = 200,
                   lr: float = 1e-4,
                   save_best: bool = True,
                   accum_steps: int = 1,
                   use_amp: bool = True):
    """
    Training loop with optional gradient accumulation and AMP.
    accum_steps > 1 increases effective batch size without increasing GPU memory.
    """
    set_seed(seed)
    model = UNet().to(device)
    criterion = HybridLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70], gamma=0.1)
    max_grad_norm = 5.0
    best_val = float('inf')

    ckpt_name = make_ckpt_name(seed)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    accum_steps = max(1, int(accum_steps))

    try:
        for epoch in range(1, num_epochs + 1):
            model.train()
            train_loss = 0.0
            # junction weight schedule
            junc_w = min(2.0, 0.3 + 1.7 * (epoch / 50))

            optimizer.zero_grad(set_to_none=True)
            for step, (imgs, masks, sks) in enumerate(train_loader):
                imgs = imgs.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.float32)
                sks = sks.to(device=device, dtype=torch.float32)

                # AMP context: use torch.amp.autocast(device_type='cuda') when running on CUDA
                if use_amp and device.type == 'cuda':
                    autocast_ctx = torch.amp.autocast(device_type='cuda', enabled=True)
                else:
                    autocast_ctx = torch.amp.autocast(device_type='cpu', enabled=False)

                with autocast_ctx:
                    logits, sk_logits = model(imgs)
                    sk_d1, sk_d2 = sk_logits

                    mask_loss = criterion(logits, masks, epoch)
                    sk_loss1 = F.binary_cross_entropy_with_logits(sk_d1, sks)
                    sk_loss2 = F.binary_cross_entropy_with_logits(sk_d2, sks)
                    junc_loss = soft_junction_loss(sk_d1, sks)
                    edge_loss = edge_consistency_loss(logits, masks)

                    loss = mask_loss + 0.6 * sk_loss1 + 0.4 * sk_loss2 + junc_w * junc_loss + 0.2 * edge_loss

                loss_scaled = loss / accum_steps
                scaler.scale(loss_scaled).backward()
                train_loss += loss.item()

                is_last_batch = (step + 1) == len(train_loader)
                if ((step + 1) % accum_steps == 0) or is_last_batch:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            scheduler.step()

            # validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, masks, sks in val_loader:
                    imgs = imgs.to(device=device, dtype=torch.float32)
                    masks = masks.to(device=device, dtype=torch.float32)
                    sks = sks.to(device=device, dtype=torch.float32)

                    logits, sk_logits = model(imgs)
                    sk_d1, sk_d2 = sk_logits

                    mask_loss = criterion(logits, masks, epoch)
                    sk_loss1 = F.binary_cross_entropy_with_logits(sk_d1, sks)
                    sk_loss2 = F.binary_cross_entropy_with_logits(sk_d2, sks)
                    junc_loss = soft_junction_loss(sk_d1, sks)
                    edge_loss = edge_consistency_loss(logits, masks)

                    val_loss += (mask_loss + 0.6 * sk_loss1 + 0.4 * sk_loss2 +
                                 junc_w * junc_loss + 0.2 * edge_loss).item()

            avg_train = train_loss / max(1, len(train_loader))
            avg_val = val_loss / max(1, len(val_loader))
            print(f"[seed {seed}] Epoch {epoch:03d}/{num_epochs} | Train {avg_train:.4f} | Val {avg_val:.4f} | LR {scheduler.get_last_lr()[0]:.1e}")

            if avg_val < best_val:
                best_val = avg_val
                if save_best:
                    torch.save(model.state_dict(), ckpt_name)
                    print(f"✅ Saved best checkpoint: {ckpt_name}")

    except KeyboardInterrupt:
        interrupt_name = f"interrupt_{make_ckpt_name(seed)}"
        torch.save(model.state_dict(), interrupt_name)
        print(f"⚠️ Interrupted. Saved: {interrupt_name}")


# -------------------------
# main: builds dataloaders and runs training for SEEDS
# -------------------------
def main():
    torch.backends.cudnn.benchmark = True

    train_img_dir = 'dataset/train/images'
    train_mask_dir = 'dataset/train/masks'
    val_img_dir = 'dataset/val/images'
    val_mask_dir = 'dataset/val/masks'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = LeafVeinDataset(train_img_dir, train_mask_dir, transform=transform, use_clahe=True)
    val_dataset = LeafVeinDataset(val_img_dir, val_mask_dir, transform=transform, use_clahe=True)

    batch_size = BATCH_SIZE
    pin = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=pin)

    device = torch.device(DEFAULT_DEVICE)
    num_epochs = 500

    for seed in SEEDS:
        print(f"\n=== Starting training run with seed {seed} ===")
        # start with accum_steps=1; increase if you want larger effective batch
        train_one_seed(seed, train_loader, val_loader, device, num_epochs=num_epochs, lr=1e-4,
                       save_best=True, accum_steps=1, use_amp=True)


if __name__ == "__main__":
    main()
