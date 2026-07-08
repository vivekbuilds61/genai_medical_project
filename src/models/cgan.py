"""
cgan.py
───────
Conditional GAN (cGAN) for medical image synthesis.

Architecture:
  Generator:    Noise z + class label → synthetic image
  Discriminator: Image + class label → real/fake score

Both use label embeddings for conditioning, enabling class-specific
image generation (e.g., generate only "Cancer Stroma" samples).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm


# ── Generator ─────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    Maps (latent_z, class_label) → RGB image of shape (3, img_size, img_size).
    Uses transposed convolutions for upsampling (more stable than pixel shuffle
    for medical textures).
    """

    def __init__(self, latent_dim=128, num_classes=9, img_channels=3, img_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size

        # Class embedding: integer label → dense vector
        self.label_emb = nn.Embedding(num_classes, latent_dim)

        # Initial projection: (latent_dim * 2) → (512 * 4 * 4)
        self.init_size = img_size // 16   # 64 → 4
        self.proj = nn.Sequential(
            nn.Linear(latent_dim * 2, 512 * self.init_size * self.init_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Upsampling blocks: 4 → 8 → 16 → 32 → 64
        self.conv_blocks = nn.Sequential(
            # Block 1: 4x4 → 8x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Block 2: 8x8 → 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Block 3: 16x16 → 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Block 4: 32x32 → 64x64
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()   # Output in [-1, 1]
        )

    def forward(self, z, labels):
        # Combine noise + label embedding
        label_emb = self.label_emb(labels)      # (B, latent_dim)
        x = torch.cat([z, label_emb], dim=1)    # (B, latent_dim*2)
        x = self.proj(x)                         # (B, 512*4*4)
        x = x.view(x.size(0), 512, self.init_size, self.init_size)
        img = self.conv_blocks(x)                # (B, 3, 64, 64)
        return img


# ── Discriminator ─────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    Maps (image, class_label) → real/fake probability.
    Label is injected as an extra channel (spatially tiled embedding).
    """

    def __init__(self, num_classes=9, img_channels=3, img_size=64):
        super().__init__()
        self.img_size = img_size

        # Label → spatial feature map (same H×W as image, 1 channel)
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)

        # Input: (img_channels + 1) channels
        self.model = nn.Sequential(
            # 64x64 → 32x32
            nn.Conv2d(img_channels + 1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32 → 16x16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 → 8x8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8 → 4x4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4 → 1x1
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Tile label embedding as spatial map
        label_map = self.label_emb(labels)                        # (B, img_size*img_size)
        label_map = label_map.view(img.size(0), 1, self.img_size, self.img_size)
        x = torch.cat([img, label_map], dim=1)                    # (B, 4, 64, 64)
        return self.model(x).view(-1)


# ── Trainer ───────────────────────────────────────────────────────────────────
class CGANTrainer:
    """
    Full training loop for the Conditional GAN.
    Includes:
      - Label smoothing for discriminator stability
      - Checkpoint saving every N epochs
      - Sample grid generation for visual progress tracking
    """

    def __init__(self, config, device=None):
        self.cfg = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[cGAN] Using device: {self.device}")

        gc = config["gan"]
        dc = config["data"]

        self.latent_dim  = gc["latent_dim"]
        self.num_classes = dc["num_classes"]
        self.img_size    = dc["image_size"]
        self.epochs      = gc["num_epochs"]
        self.smooth      = gc["label_smoothing"]

        # Models
        self.G = Generator(self.latent_dim, self.num_classes, 3, self.img_size).to(self.device)
        self.D = Discriminator(self.num_classes, 3, self.img_size).to(self.device)

        # Weight initialisation (DCGAN-style)
        self.G.apply(self._weights_init)
        self.D.apply(self._weights_init)

        # Optimizers
        self.opt_G = optim.Adam(self.G.parameters(), lr=gc["lr_generator"],     betas=(gc["beta1"], gc["beta2"]))
        self.opt_D = optim.Adam(self.D.parameters(), lr=gc["lr_discriminator"], betas=(gc["beta1"], gc["beta2"]))

        self.criterion = nn.BCELoss()
        self.history   = {"g_loss": [], "d_loss": []}

        os.makedirs(config["paths"]["models_gan"], exist_ok=True)
        os.makedirs(config["paths"]["data_synthetic"], exist_ok=True)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def _sample_z(self, n):
        return torch.randn(n, self.latent_dim, device=self.device)

    def train(self, dataloader):
        print(f"\n[cGAN] Training for {self.epochs} epochs …\n")
        for epoch in range(1, self.epochs + 1):
            g_losses, d_losses = [], []

            for real_imgs, labels in tqdm(dataloader, desc=f"Epoch {epoch}/{self.epochs}", leave=False):
                real_imgs = real_imgs.to(self.device)
                labels    = labels.to(self.device)
                B = real_imgs.size(0)

                # ── Train Discriminator ──────────────────────────
                self.opt_D.zero_grad()
                real_labels = torch.ones(B, device=self.device)  * (1 - self.smooth)
                fake_labels = torch.zeros(B, device=self.device) + self.smooth

                # Real
                d_real = self.D(real_imgs, labels)
                loss_real = self.criterion(d_real, real_labels)

                # Fake
                z         = self._sample_z(B)
                rand_lbls = torch.randint(0, self.num_classes, (B,), device=self.device)
                fake_imgs = self.G(z, rand_lbls).detach()
                d_fake    = self.D(fake_imgs, rand_lbls)
                loss_fake = self.criterion(d_fake, fake_labels)

                d_loss = (loss_real + loss_fake) / 2
                d_loss.backward()
                self.opt_D.step()

                # ── Train Generator ──────────────────────────────
                self.opt_G.zero_grad()
                z         = self._sample_z(B)
                rand_lbls = torch.randint(0, self.num_classes, (B,), device=self.device)
                fake_imgs = self.G(z, rand_lbls)
                d_fake    = self.D(fake_imgs, rand_lbls)
                g_loss    = self.criterion(d_fake, torch.ones(B, device=self.device))
                g_loss.backward()
                self.opt_G.step()

                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

            avg_g = np.mean(g_losses)
            avg_d = np.mean(d_losses)
            self.history["g_loss"].append(avg_g)
            self.history["d_loss"].append(avg_d)
            print(f"  Epoch {epoch:3d} | G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f}")

            # Save checkpoint
            if epoch % self.cfg["gan"]["save_every"] == 0:
                self._save_checkpoint(epoch)

        print("[cGAN] Training complete.")
        return self.history

    def _save_checkpoint(self, epoch):
        path = os.path.join(self.cfg["paths"]["models_gan"], f"checkpoint_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "G_state": self.G.state_dict(),
            "D_state": self.D.state_dict(),
            "opt_G":   self.opt_G.state_dict(),
            "opt_D":   self.opt_D.state_dict(),
            "history": self.history
        }, path)
        print(f"  [✓] Checkpoint saved → {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G_state"])
        self.D.load_state_dict(ckpt["D_state"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D.load_state_dict(ckpt["opt_D"])
        self.history = ckpt["history"]
        print(f"[cGAN] Loaded checkpoint from epoch {ckpt['epoch']}")

    def generate_synthetic_dataset(self, num_images=5000, save=True):
        """
        Generate a balanced synthetic dataset across all classes.
        Returns numpy arrays (images, labels).
        """
        self.G.eval()
        per_class = num_images // self.num_classes
        all_imgs, all_labels = [], []

        with torch.no_grad():
            for cls_idx in range(self.num_classes):
                z      = self._sample_z(per_class)
                labels = torch.full((per_class,), cls_idx, dtype=torch.long, device=self.device)
                imgs   = self.G(z, labels).cpu().numpy()   # (N, 3, H, W) in [-1, 1]
                all_imgs.append(imgs)
                all_labels.extend([cls_idx] * per_class)

        all_imgs   = np.concatenate(all_imgs, axis=0).astype(np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)

        if save:
            np.save(os.path.join(self.cfg["paths"]["data_synthetic"], "synthetic_images.npy"), all_imgs)
            np.save(os.path.join(self.cfg["paths"]["data_synthetic"], "synthetic_labels.npy"), all_labels)
            print(f"[cGAN] Saved {len(all_imgs)} synthetic images.")

        self.G.train()
        return all_imgs, all_labels
