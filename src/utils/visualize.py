"""
visualize.py
────────────
All visualisation utilities for the pipeline:
  - GAN training loss curves
  - Synthetic image grids
  - CNN training curves + confusion matrix
  - Biomarker feature t-SNE / UMAP
  - Drug candidate bar charts
  - Molecular structure rendering
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# Try to import RDKit drawing — optional for structure plots
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_DRAW = True
except ImportError:
    RDKIT_DRAW = False

PATHMNIST_CLASSES = [
    "Adipose", "Background", "Debris", "Lymphocytes",
    "Mucus", "Smooth Muscle", "Normal Colon",
    "Cancer Stroma", "Colorectal Adenocarcinoma"
]

# Consistent colour palette
PALETTE = ["#00e5a0", "#0099ff", "#ff6b35", "#a855f7",
           "#f59e0b", "#ec4899", "#14b8a6", "#6366f1", "#84cc16"]

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#111820",
    "axes.edgecolor":   "#2d3748",
    "axes.labelcolor":  "#8b99a8",
    "xtick.color":      "#8b99a8",
    "ytick.color":      "#8b99a8",
    "text.color":       "#e8edf2",
    "grid.color":       "#1f2937",
    "grid.alpha":       0.5,
    "font.family":      "monospace",
})


# ── 1. GAN Loss Curves ────────────────────────────────────────────────────────
def plot_gan_losses(history: dict, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    epochs = range(1, len(history["g_loss"]) + 1)
    ax.plot(epochs, history["g_loss"], color=PALETTE[0],
            lw=2, label="Generator Loss")
    ax.plot(epochs, history["d_loss"], color=PALETTE[1],
            lw=2, label="Discriminator Loss")
    ax.fill_between(epochs, history["g_loss"], alpha=0.08, color=PALETTE[0])
    ax.fill_between(epochs, history["d_loss"], alpha=0.08, color=PALETTE[1])
    ax.set_title("cGAN Training Loss", fontsize=14, color="#e8edf2", pad=12)
    ax.set_xlabel("Epoch");  ax.set_ylabel("Loss")
    ax.legend(framealpha=0.2)
    ax.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show();  plt.close()


# ── 2. Synthetic Image Grid ───────────────────────────────────────────────────
def plot_synthetic_grid(images: np.ndarray, labels: np.ndarray,
                        n_per_class=4, save_path=None):
    """
    images: (N, 3, H, W) float32 in [-1, 1]
    labels: (N,) int
    """
    n_classes = len(np.unique(labels))
    fig, axes = plt.subplots(n_classes, n_per_class,
                             figsize=(n_per_class * 2, n_classes * 2))
    fig.suptitle("GAN-Generated Synthetic Medical Images", fontsize=13,
                 color="#e8edf2", y=1.01)

    for cls_idx in range(n_classes):
        cls_imgs = images[labels == cls_idx][:n_per_class]
        for col, img in enumerate(cls_imgs):
            ax = axes[cls_idx, col]
            # Denormalise [-1,1] → [0,1]
            img_disp = (img.transpose(1, 2, 0) + 1) / 2
            img_disp = np.clip(img_disp, 0, 1)
            ax.imshow(img_disp)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(PATHMNIST_CLASSES[cls_idx] if cls_idx < len(PATHMNIST_CLASSES) else str(cls_idx),
                              fontsize=7, color="#e8edf2", rotation=0, labelpad=80, va="center")

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show();  plt.close()


# ── 3. CNN Training Curves ────────────────────────────────────────────────────
def plot_cnn_training(history: dict, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"],
             color=PALETTE[0], lw=2, label="Train")
    ax1.plot(epochs, history["val_loss"],
             color=PALETTE[1], lw=2, label="Val", ls="--")
    ax1.set_title("Loss", fontsize=12, color="#e8edf2")
    ax1.set_xlabel("Epoch");  ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(framealpha=0.2);  ax1.grid(True, ls="--", lw=0.5)

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]],
             color=PALETTE[0], lw=2, label="Train")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]],
             color=PALETTE[1], lw=2, label="Val", ls="--")
    ax2.set_title("Accuracy", fontsize=12, color="#e8edf2")
    ax2.set_xlabel("Epoch");  ax2.set_ylabel("Accuracy (%)")
    ax2.legend(framealpha=0.2);  ax2.grid(True, ls="--", lw=0.5)

    fig.suptitle("CNN Classifier Training Curves",
                 fontsize=13, color="#e8edf2")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show();  plt.close()


# ── 4. Confusion Matrix ───────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, class_names=None, save_path=None):
    if class_names is None:
        class_names = PATHMNIST_CLASSES

    # Normalise
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5, linecolor="#1f2937",
                cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_title("CNN Confusion Matrix (Normalised)",
                 fontsize=13, color="#e8edf2", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show();  plt.close()


# ── 5. Feature t-SNE ─────────────────────────────────────────────────────────
def plot_tsne(features: np.ndarray, labels: np.ndarray,
              title="Biomarker Feature Space (t-SNE)", save_path=None):
    print("[Viz] Running t-SNE (this may take ~30s) …")

    # PCA first to speed up t-SNE
    n_pca = min(50, features.shape[1], features.shape[0] - 1)
    pca_feats = PCA(n_components=n_pca,
                    random_state=42).fit_transform(features)

    tsne = TSNE(n_components=2, perplexity=40,
                n_iter=1000, random_state=42, verbose=0)
    emb = tsne.fit_transform(pca_feats)

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls_idx in np.unique(labels):
        mask = labels == cls_idx
        label_name = PATHMNIST_CLASSES[cls_idx] if cls_idx < len(
            PATHMNIST_CLASSES) else str(cls_idx)
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=PALETTE[cls_idx % len(PALETTE)],
                   label=label_name, alpha=0.6, s=18, edgecolors="none")

    ax.set_title(title, fontsize=13, color="#e8edf2", pad=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
              framealpha=0.2, markerscale=1.5)
    ax.set_xlabel("t-SNE 1");  ax.set_ylabel("t-SNE 2")
    ax.grid(True, ls="--", lw=0.5, alpha=0.4)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show();  plt.close()


# ── 6. Drug Candidate Bar Chart ───────────────────────────────────────────────
def plot_drug_candidates(results: dict, class_name: str, save_path=None):
    df = results.get(class_name)
    if df is None or len(df) == 0:
        print(f"[Viz] No results for class: {class_name}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [PALETTE[0] if r > 0 else PALETTE[2] for r in df["correlation"]]
    bars = ax.barh(df["name"], df["abs_corr"], color=colors,
                   edgecolor="none", height=0.6)

        # Annotate with correlation value
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"r={row['correlation']:+.3f} [{row['target']}]",
            va="center",
            fontsize=8.5,
            color="#e8edf2",
        )

    ax.set_xlim(0, df["abs_corr"].max() + 0.12)
    ax.set_title(
        f"Top Drug Candidates - {class_name}",
        fontsize=13,
        color="#e8edf2",
        pad=12,
    )
    ax.set_xlabel("|Pearson Correlation|")
    ax.invert_yaxis()
    ax.grid(True, axis="x", ls="--", lw=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


# ── 7. Multi-class Drug Heatmap ───────────────────────────────────────────────
def plot_correlation_heatmap(summary_df, save_path=None):
    """Heatmap: drugs × disease classes coloured by |correlation|."""

    pivot = summary_df.pivot_table(
        index="Drug",
        columns="Disease Class",
        values="Abs Corr",
        aggfunc="mean",
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        pivot,
        cmap="YlGnBu",
        ax=ax,
        linewidths=0.3,
        linecolor="#1f2937",
        cbar_kws={"shrink": 0.7},
    )

    ax.set_title(
        "Drug-Disease Biomarker Correlation Heatmap",
        fontsize=13,
        color="#e8edf2",
        pad=12,
    )

    ax.set_xlabel("Disease Class", fontsize=10)
    ax.set_ylabel("Drug Compound", fontsize=10)

    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


# ── 8. Molecular Structure Grid ───────────────────────────────────────────────
def plot_molecule_structures(smiles_list: list, names: list, save_path=None):
    """Plot molecular structures using RDKit."""

    if not RDKIT_DRAW:
        print("[Viz] RDKit Draw not available - skipping structure plot.")
        return

    mols = []
    legends = []

    for smile, name in zip(smiles_list, names):
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            mols.append(mol)
            legends.append(name)

    img = Draw.MolsToGridImage(
        mols[:12],
        molsPerRow=4,
        subImgSize=(300, 200),
        legends=legends[:12],
    )

    if save_path:
        img.save(save_path)
    else:
        img.show()