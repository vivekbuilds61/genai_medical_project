"""
cnn_encoder.py
──────────────
ResNet-18 based CNN that does two things:
  1. Classifies medical images into 9 tissue classes
  2. Extracts a 256-dim biomarker feature vector per image
     (used downstream for drug correlation analysis)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import torchvision.models as tv_models


# ── Model ─────────────────────────────────────────────────────────────────────
class CNNEncoder(nn.Module):
    """
    ResNet-18 backbone with:
      - Custom classification head (9 classes)
      - Separate feature extraction head (256-dim biomarker vector)
    """

    def __init__(self, num_classes=9, feature_dim=256, dropout=0.4, pretrained=True):
        super().__init__()

        # Load ResNet-18 backbone
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        # Remove the final FC layer — keep everything up to avgpool
        self.backbone = nn.Sequential(
            *list(backbone.children())[:-1])  # Output: (B, 512, 1, 1)

        # Biomarker feature head — maps 512 → 256
        self.feature_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Classification head — maps 256 → num_classes
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, return_features=False):
        """
        Args:
            x: (B, 3, H, W) image tensor
            return_features: if True, also return 256-dim biomarker vector
        Returns:
            logits: (B, num_classes)
            features (optional): (B, feature_dim)
        """
        backbone_out = self.backbone(x)          # (B, 512, 1, 1)
        features = self.feature_head(backbone_out)  # (B, 256)
        logits = self.classifier(features)        # (B, 9)

        if return_features:
            return logits, features
        return logits

    def extract_features(self, x):
        """Convenience: return only the 256-dim biomarker vector."""
        with torch.no_grad():
            backbone_out = self.backbone(x)
            features = self.feature_head(backbone_out)
        return features


# ── Trainer ───────────────────────────────────────────────────────────────────
class CNNTrainer:
    """
    Full training + evaluation loop for the CNN classifier.
    Includes cosine LR scheduling, best-model checkpointing,
    and feature extraction for the full dataset.
    """

    def __init__(self, config, device=None):
        self.cfg = config
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CNN] Using device: {self.device}")

        cc = config["cnn"]
        dc = config["data"]

        self.num_classes = dc["num_classes"]
        self.feature_dim = cc["feature_dim"]
        self.epochs = cc["num_epochs"]

        self.model = CNNEncoder(
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            dropout=cc["dropout"],
            pretrained=True
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cc["lr"],
            weight_decay=cc["weight_decay"]
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6)

        self.history = {"train_loss": [], "val_loss": [],
                        "train_acc": [], "val_acc": []}
        self.best_acc = 0.0

        os.makedirs(config["paths"]["models_cnn"], exist_ok=True)

    def _run_epoch(self, loader, training=True):
        self.model.train() if training else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for imgs, labels in tqdm(loader, leave=False, desc="Train" if training else "Val"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                if training:
                    self.optimizer.zero_grad()

                logits = self.model(imgs)
                loss = self.criterion(logits, labels)

                if training:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += imgs.size(0)

        return total_loss / total, correct / total

    def train(self, train_loader, val_loader):
        print(f"\n[CNN] Training for {self.epochs} epochs …\n")
        for epoch in range(1, self.epochs + 1):
            tr_loss, tr_acc = self._run_epoch(train_loader, training=True)
            va_loss, va_acc = self._run_epoch(val_loader,   training=False)
            self.scheduler.step()

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_acc"].append(va_acc)

            flag = ""
            if va_acc > self.best_acc:
                self.best_acc = va_acc
                self._save_best()
                flag = "  ← best"

            print(f"  Epoch {epoch:3d}/{self.epochs} | "
                  f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | "
                  f"Val Loss: {va_loss:.4f} Acc: {va_acc:.3f}{flag}")

        print(f"\n[CNN] Training complete. Best Val Acc: {self.best_acc:.4f}")
        return self.history

    def _save_best(self):
        path = os.path.join(self.cfg["paths"]["models_cnn"], "best_cnn.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "best_acc":    self.best_acc,
            "history":     self.history
        }, path)

    def load_best(self):
        path = os.path.join(self.cfg["paths"]["models_cnn"], "best_cnn.pt")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.best_acc = ckpt["best_acc"]
        print(f"[CNN] Loaded best model (Val Acc: {self.best_acc:.4f})")

    def evaluate(self, test_loader, class_names=None):
        """Full evaluation: accuracy + classification report + confusion matrix."""
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
                imgs = imgs.to(self.device)
                logits = self.model(imgs)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = (all_preds == all_labels).mean()

        print(f"\n[CNN] Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        if class_names:
            print(classification_report(all_labels,
                  all_preds, target_names=class_names))

        cm = confusion_matrix(all_labels, all_preds)
        return accuracy, all_preds, all_labels, cm

    def extract_dataset_features(self, loader):
        """
        Extract 256-dim biomarker feature vectors for an entire dataset.
        Returns:
            features: np.ndarray (N, 256)
            labels:   np.ndarray (N,)
        """
        self.model.eval()
        all_features, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc="Extracting features", leave=False):
                imgs = imgs.to(self.device)
                _, feats = self.model(imgs, return_features=True)
                all_features.append(feats.cpu().numpy())
                all_labels.extend(labels.numpy())

        features = np.concatenate(all_features, axis=0)
        labels = np.array(all_labels)

        # Save for reuse
        save_dir = self.cfg["paths"]["data_processed"]
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "biomarker_features.npy"), features)
        np.save(os.path.join(save_dir, "biomarker_labels.npy"),   labels)

        print(
            f"[CNN] Extracted features: {features.shape}  →  saved to {save_dir}")
        return features, labels
