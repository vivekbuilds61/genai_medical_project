"""
dataset.py
──────────
Loaders for:
  1. MedMNIST PathMNIST  — 9-class colon pathology images
  2. Molecular dataset   — curated SMILES + simulated bioactivity
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import medmnist
from medmnist import PathMNIST
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import yaml
from pathlib import Path


# ── Load config ──────────────────────────────────────────────────────────────
def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Class labels for PathMNIST ────────────────────────────────────────────────
PATHMNIST_CLASSES = [
    "Adipose", "Background", "Debris", "Lymphocytes",
    "Mucus", "Smooth Muscle", "Normal Colon", "Cancer Stroma", "Colorectal Adenocarcinoma"
]


# ── 1. Medical Image Dataset ──────────────────────────────────────────────────
class MedicalImageDataset(Dataset):
    """
    Wraps MedMNIST PathMNIST with configurable transforms.
    Labels are integers 0–8 (9 tissue classes).
    """

    def __init__(self, split="train", image_size=64, augment=False, download=True, data_root="data/raw"):
        os.makedirs(data_root, exist_ok=True)

        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        if augment and split == "train":
            transform_list = [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]

        self.transform = transforms.Compose(transform_list)
        self.dataset = PathMNIST(
            split=split,
            transform=self.transform,
            download=download,
            root=data_root,
            size=image_size
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label.squeeze().long()

    @property
    def num_classes(self):
        return 9


def get_dataloaders(config_path="configs/config.yaml", download=True):
    """Return train, val, test DataLoaders for PathMNIST."""
    cfg = load_config(config_path)
    dc = cfg["data"]

    train_ds = MedicalImageDataset("train", dc["image_size"], augment=dc["augmentation"] if "augmentation" in dc else True, download=download, data_root=cfg["paths"]["data_raw"])
    val_ds   = MedicalImageDataset("val",   dc["image_size"], augment=False, download=download, data_root=cfg["paths"]["data_raw"])
    test_ds  = MedicalImageDataset("test",  dc["image_size"], augment=False, download=download, data_root=cfg["paths"]["data_raw"])

    train_loader = DataLoader(train_ds, batch_size=dc["batch_size"], shuffle=True,  num_workers=dc["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=dc["batch_size"], shuffle=False, num_workers=dc["num_workers"])
    test_loader  = DataLoader(test_ds,  batch_size=dc["batch_size"], shuffle=False, num_workers=dc["num_workers"])

    print(f"[Dataset] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


# ── 2. Synthetic Dataset (GAN-generated images) ───────────────────────────────
class SyntheticImageDataset(Dataset):
    """Loads GAN-generated synthetic images saved as .npy arrays."""

    def __init__(self, images_path, labels_path, transform=None):
        self.images = np.load(images_path)   # Shape: (N, 3, H, W) float32 in [-1, 1]
        self.labels = np.load(labels_path)   # Shape: (N,) int
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32)
        lbl = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.transform:
            img = self.transform(img)
        return img, lbl


# ── 3. Molecular Dataset ──────────────────────────────────────────────────────
# Curated SMILES of well-known compounds with annotated targets.
# In a real project these would come from ChEMBL / PubChem APIs.
CURATED_MOLECULES = [
    # (name, SMILES, target, bioactivity_IC50_nM)
    ("Ibuprofen",     "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                        "COX-2",     13000),
    ("Aspirin",       "CC(=O)Oc1ccccc1C(=O)O",                              "COX-1",     50000),
    ("Metformin",     "CN(C)C(=N)NC(=N)N",                                  "AMPK",       1200),
    ("Erlotinib",     "C#Cc1cccc(Nc2ncnc3cc(OCC)c(OCC)cc23)c1",             "EGFR",         2),
    ("Imatinib",      "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1", "BCR-ABL", 100),
    ("Gefitinib",     "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",    "EGFR",         33),
    ("Sorafenib",     "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1","RAF",  10),
    ("Vemurafenib",   "CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3cc(ccc23)-c2ccc(Cl)cc2)c1","BRAF", 31),
    ("Tamoxifen",     "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",       "ESR1",       3400),
    ("Doxorubicin",   "COc1cccc2C(=O)c3c(O)c4C(O)(CC(O)(CC(=O)CO)c4c(O)c3=O)c21", "TOP2A", 200),
    ("Methotrexate",  "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(cc1)C(=O)NC(CCC(=O)O)C(=O)O","DHFR", 5),
    ("Cisplatin",     "[NH3][Pt]([NH3])(Cl)Cl",                             "DNA",        3000),
    ("Paclitaxel",    "O=C(O[C@@H]1C[C@]2(OC(=O)c3ccccc3)[C@@H](O)C[C@@H](O)[C@H]3[C@H](OC(=O)[C@@H](O)[C@@H](NC(=O)c4ccccc4)[C@@H](O)c4ccccc4)C(=O)[C@@](C)(O)[C@@H]3[C@@H]2OC(C)=O)c1C","Tubulin", 3),
    ("Carboplatin",   "O=C1OC(=O)[C@H]2CCCCC[Pt]12",                       "DNA",        1000),
    ("Vincristine",   "CCC1(CC)C=C2CN3CCc4cc5c(cc4[C@@H]3C[C@H]2C1)OC(=O)c1[nH]cc(CC)c1CC", "Tubulin", 50),
    ("Thalidomide",   "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",               "CRBN",       100),
    ("Bortezomib",    "CC(C)C[C@@H](NC(=O)[C@@H](Cc1cccnc1)NC(=O)c1cnccn1)B(O)O", "26S-Proteasome", 0.6),
    ("Lenalidomide",  "O=C1CCC(N2C(=O)c3cccc(N)c3C2=O)C(=O)N1",            "CRBN",       1000),
    ("Ibrutinib",     "O=C(/C=C/c1ccccc1)N1CCC[C@@H]1c1ncnc2[nH]ccc12",   "BTK",          0.5),
    ("Ruxolitinib",   "C[C@@H](Cc1cncn1C)Nc1ncnc2[nH]cc(-c3ccncc3)c12",    "JAK1/2",      3.3),
]


class MolecularDataset:
    """
    Computes molecular descriptors using RDKit for a curated compound set.
    Returns a DataFrame with SMILES, descriptors, and bioactivity labels.
    """

    def __init__(self, descriptor_type="morgan", morgan_radius=2, morgan_bits=2048):
        self.descriptor_type = descriptor_type
        self.morgan_radius = morgan_radius
        self.morgan_bits = morgan_bits

    def compute_descriptors(self, smiles: str):
        """Return a numpy feature vector for a given SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if self.descriptor_type == "morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.morgan_radius, self.morgan_bits)
            return np.array(fp)
        elif self.descriptor_type == "rdkit":
            desc_names = [d[0] for d in Descriptors.descList[:50]]
            desc_vals = []
            for name in desc_names:
                try:
                    val = getattr(Descriptors, name)(mol)
                    desc_vals.append(val if np.isfinite(val) else 0.0)
                except:
                    desc_vals.append(0.0)
            return np.array(desc_vals, dtype=np.float32)
        else:
            from rdkit.Chem import MACCSkeys
            fp = MACCSkeys.GenMACCSKeys(mol)
            return np.array(fp)

    def build_dataframe(self):
        """Build a clean DataFrame with all descriptors computed."""
        records = []
        for name, smiles, target, ic50 in CURATED_MOLECULES:
            desc = self.compute_descriptors(smiles)
            if desc is not None:
                records.append({
                    "name": name,
                    "smiles": smiles,
                    "target": target,
                    "ic50_nM": ic50,
                    "pIC50": -np.log10(ic50 * 1e-9),   # Convert to pIC50
                    "descriptors": desc
                })
        df = pd.DataFrame(records)
        print(f"[Molecular] Loaded {len(df)} molecules with {len(df.iloc[0]['descriptors'])}-dim descriptors")
        return df
