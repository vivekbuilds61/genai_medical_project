"""
metrics.py
──────────
Evaluation metrics for the full pipeline:
  - FID Score (Frechet Inception Distance) for GAN quality
  - Classification metrics (accuracy, F1, AUC)
  - Biomarker correlation quality metrics
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score
)
from torchvision.models import inception_v3, Inception_V3_Weights
import warnings
warnings.filterwarnings("ignore")


# ── 1. FID Score ──────────────────────────────────────────────────────────────
class FIDCalculator:
    """
    Frechet Inception Distance between real and synthetic image distributions.
    Lower FID = better quality synthetic images.
    Typical good values: < 50 for medical images.
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load Inception v3 for feature extraction
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()   # Remove final classifier
        self.model.eval().to(self.device)
        print(f"[FID] Inception v3 loaded on {self.device}")

    def _get_activations(self, images_np: np.ndarray, batch_size=64) -> np.ndarray:
        """
        images_np: (N, 3, H, W) float32 in [-1, 1]
        Returns:   (N, 2048) Inception features
        """
        import torch.nn.functional as F

        activations = []
        n = len(images_np)

        for start in range(0, n, batch_size):
            batch = torch.tensor(images_np[start:start + batch_size], dtype=torch.float32)
            # Inception expects (B, 3, 299, 299)
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
            # Denormalise [-1,1] → [0,1] → inception range
            batch = (batch + 1) / 2
            batch = batch.to(self.device)

            with torch.no_grad():
                feats = self.model(batch)
                if isinstance(feats, tuple):
                    feats = feats[0]
            activations.append(feats.cpu().numpy())

        return np.concatenate(activations, axis=0)

    @staticmethod
    def _compute_statistics(activations: np.ndarray):
        mu  = activations.mean(axis=0)
        cov = np.cov(activations, rowvar=False)
        return mu, cov

    @staticmethod
    def _frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(cov1 @ cov2, disp=False)

        if not np.isfinite(covmean).all():
            offset  = np.eye(cov1.shape[0]) * eps
            covmean = linalg.sqrtm((cov1 + offset) @ (cov2 + offset))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff @ diff + np.trace(cov1 + cov2 - 2 * covmean)
        return float(fid)

    def compute(self, real_images: np.ndarray, fake_images: np.ndarray) -> float:
        """
        Compute FID between real and synthetic image sets.
        Both should be (N, 3, H, W) float32 in [-1, 1].
        Minimum 2048 images recommended for reliable FID.
        """
        print(f"[FID] Computing activations for {len(real_images)} real + {len(fake_images)} synthetic images …")
        real_acts = self._get_activations(real_images)
        fake_acts = self._get_activations(fake_images)

        mu_r, cov_r = self._compute_statistics(real_acts)
        mu_f, cov_f = self._compute_statistics(fake_acts)

        fid = self._frechet_distance(mu_r, cov_r, mu_f, cov_f)
        print(f"[FID] Score: {fid:.2f}  (lower is better; <50 is good for medical images)")
        return fid


# ── 2. Classification Metrics ─────────────────────────────────────────────────
def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    class_names: list = None
) -> dict:
    """
    Compute a comprehensive set of classification metrics.

    Returns dict with accuracy, macro F1, weighted F1, per-class F1,
    and optionally AUC-ROC and AUC-PR.
    """
    results = {
        "accuracy":   accuracy_score(y_true, y_pred),
        "f1_macro":   f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted":f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_per_class": f1_score(y_true, y_pred, average=None,     zero_division=0).tolist()
    }

    if y_proba is not None:
        try:
            results["auc_roc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            results["auc_pr"]  = average_precision_score(
                np.eye(y_proba.shape[1])[y_true], y_proba, average="macro"
            )
        except Exception:
            pass

    # Print summary
    print(f"\n[Metrics] Accuracy:     {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"[Metrics] F1 Macro:     {results['f1_macro']:.4f}")
    print(f"[Metrics] F1 Weighted:  {results['f1_weighted']:.4f}")
    if "auc_roc" in results:
        print(f"[Metrics] AUC-ROC:      {results['auc_roc']:.4f}")

    if class_names:
        print("\n[Metrics] Per-class F1:")
        for name, f1 in zip(class_names, results["f1_per_class"]):
            bar = "█" * int(f1 * 20)
            print(f"  {name:30s} {bar:<20s} {f1:.3f}")

    return results


# ── 3. Correlation Quality Metrics ────────────────────────────────────────────
def compute_correlation_summary(results: dict) -> dict:
    """
    Summarise the drug correlation results across all disease classes.

    Returns dict with mean/max correlations, significant hit counts, etc.
    """
    summary = {}
    for class_name, df in results.items():
        sig_hits  = df["significant"].sum() if "significant" in df.columns else 0
        mean_corr = df["abs_corr"].mean()   if "abs_corr"    in df.columns else 0
        max_corr  = df["abs_corr"].max()    if "abs_corr"    in df.columns else 0
        top_drug  = df.iloc[0]["name"]      if len(df) > 0 else "N/A"

        summary[class_name] = {
            "mean_abs_corr":  round(float(mean_corr), 4),
            "max_abs_corr":   round(float(max_corr),  4),
            "significant_hits": int(sig_hits),
            "top_candidate":  top_drug
        }

    # Overall stats
    all_mean = np.mean([v["mean_abs_corr"] for v in summary.values()])
    all_max  = np.max([v["max_abs_corr"]   for v in summary.values()])
    total_sig = sum(v["significant_hits"]  for v in summary.values())

    print(f"\n[Correlation] Mean |r| across all classes: {all_mean:.4f}")
    print(f"[Correlation] Max  |r| observed:           {all_max:.4f}")
    print(f"[Correlation] Total significant hits (p<0.05): {total_sig}")

    summary["_overall"] = {
        "mean_abs_corr": round(float(all_mean), 4),
        "max_abs_corr":  round(float(all_max),  4),
        "total_significant_hits": total_sig
    }
    return summary


# ── 4. GAN Training Health Check ──────────────────────────────────────────────
def check_gan_health(history: dict) -> dict:
    """
    Diagnose common GAN training pathologies from loss history.
    Returns dict with mode collapse risk, convergence status, etc.
    """
    g_losses = np.array(history["g_loss"])
    d_losses = np.array(history["d_loss"])

    # Mode collapse: G loss increases sharply while D loss drops near 0
    d_min      = d_losses.min()
    g_variance = g_losses[-10:].std() if len(g_losses) >= 10 else g_losses.std()

    mode_collapse_risk = "HIGH"   if d_min < 0.05 else \
                         "MEDIUM" if d_min < 0.15 else "LOW"

    # Convergence: G and D losses stabilising in last 20% of training
    last_20pct = max(1, len(g_losses) // 5)
    g_stable   = g_losses[-last_20pct:].std() < 0.1
    d_stable   = d_losses[-last_20pct:].std() < 0.1
    converged  = g_stable and d_stable

    health = {
        "mode_collapse_risk": mode_collapse_risk,
        "converged":          converged,
        "final_g_loss":       round(float(g_losses[-1]), 4),
        "final_d_loss":       round(float(d_losses[-1]), 4),
        "g_loss_variance":    round(float(g_variance),   4),
    }

    print(f"\n[GAN Health] Mode Collapse Risk: {mode_collapse_risk}")
    print(f"[GAN Health] Converged:          {converged}")
    print(f"[GAN Health] Final G/D Loss:     {health['final_g_loss']} / {health['final_d_loss']}")
    return health
