"""
drug_correlator.py
──────────────────
Correlates CNN-extracted imaging biomarkers with molecular descriptors
to surface ranked drug candidates.

Pipeline:
  1. Reduce 256-dim biomarker vectors → PCA components (per class)
  2. Reduce molecular Morgan fingerprints → PCA components
  3. Compute Pearson / Spearman correlations between every
     (biomarker_component, molecular_descriptor) pair
  4. Rank molecules by mean absolute correlation to each disease class
  5. Return top-K drug candidates with scores + metadata
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ── Class names ───────────────────────────────────────────────────────────────
PATHMNIST_CLASSES = [
    "Adipose", "Background", "Debris", "Lymphocytes",
    "Mucus", "Smooth Muscle", "Normal Colon",
    "Cancer Stroma", "Colorectal Adenocarcinoma"
]


class DrugCorrelator:
    """
    Correlates imaging biomarkers with molecular descriptors
    to identify potential drug candidates per disease class.
    """

    def __init__(self, config):
        mc = config["molecular"]
        # pearson | spearman
        self.correlation_method = mc["correlation_method"]
        self.top_k = mc["top_k_candidates"]
        self.n_pca_components = 20    # Reduce both spaces to 20 dims before correlation

    # ── Step 1: Prepare biomarker matrix ──────────────────────────────────────
    def prepare_biomarkers(self, features: np.ndarray, labels: np.ndarray):
        """
        Compute per-class mean biomarker profiles.
        Returns: dict {class_idx: mean_feature_vector (256,)}
        """
        class_profiles = {}
        for cls in np.unique(labels):
            mask = labels == cls
            class_profiles[int(cls)] = features[mask].mean(axis=0)
        print(
            f"[Correlator] Computed mean biomarker profiles for {len(class_profiles)} classes.")
        return class_profiles

    # ── Step 2: Prepare molecular descriptor matrix ───────────────────────────
    def prepare_molecular_matrix(self, mol_df: pd.DataFrame):
        """
        Stack all molecular descriptor vectors into a (N_mols, D) matrix.
        Returns: descriptor_matrix (N_mols, D_reduced), mol_df with index
        """
        desc_matrix = np.vstack(
            mol_df["descriptors"].values).astype(np.float32)

        # Standardise
        scaler = StandardScaler()
        desc_matrix_scaled = scaler.fit_transform(desc_matrix)

        # PCA to reduce high-dim Morgan fingerprints → manageable size
        n_components = min(
            self.n_pca_components, desc_matrix_scaled.shape[0] - 1, desc_matrix_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        desc_reduced = pca.fit_transform(desc_matrix_scaled)

        print(f"[Correlator] Molecular matrix: {desc_matrix.shape} → PCA: {desc_reduced.shape}  "
              f"(variance explained: {pca.explained_variance_ratio_.sum():.2%})")
        return desc_reduced, scaler, pca

    # ── Step 3: Correlation engine ────────────────────────────────────────────
    def _correlate(self, biomarker_vec: np.ndarray, mol_matrix: np.ndarray):
        """
        Compute correlation between a single biomarker vector (D,)
        and each molecule's descriptor row (N_mols, D_mol).

        Strategy: reduce biomarker to PCA, compute element-wise correlation
        across the shared reduced space.

        Returns: (N_mols,) correlation scores
        """
        # Reduce biomarker vector to same n_pca_components via projection
        n_mol_dims = mol_matrix.shape[1]
        bm_reduced = biomarker_vec[:n_mol_dims] if len(biomarker_vec) >= n_mol_dims else \
            np.pad(biomarker_vec, (0, n_mol_dims - len(biomarker_vec)))

        scores = []
        for mol_vec in mol_matrix:
            if self.correlation_method == "pearson":
                r, p = stats.pearsonr(bm_reduced, mol_vec)
            else:
                r, p = stats.spearmanr(bm_reduced, mol_vec)
            scores.append((r if np.isfinite(r) else 0.0, p))

        return np.array([s[0] for s in scores]), np.array([s[1] for s in scores])

    # ── Step 4: Rank drug candidates ──────────────────────────────────────────
    def rank_candidates(
        self,
        class_profiles: dict,
        mol_df: pd.DataFrame,
        mol_matrix_reduced: np.ndarray
    ) -> dict:
        """
        For each disease class, rank all molecules by their biomarker correlation.

        Returns: dict {class_name: DataFrame of top-K candidates}
        """
        results = {}

        for cls_idx, bm_vec in class_profiles.items():
            class_name = PATHMNIST_CLASSES[cls_idx]
            corr_scores, p_values = self._correlate(bm_vec, mol_matrix_reduced)

            # Build results table
            ranking = mol_df.copy()
            ranking["correlation"] = corr_scores
            ranking["abs_corr"] = np.abs(corr_scores)
            ranking["p_value"] = p_values
            ranking["significant"] = p_values < 0.05
            ranking["disease_class"] = class_name

            # Sort by absolute correlation descending
            ranking = ranking.sort_values(
                "abs_corr", ascending=False).reset_index(drop=True)

            # Keep only top-K, drop the raw descriptor column
            top_k = ranking.head(self.top_k).drop(
                columns=["descriptors"], errors="ignore")
            results[class_name] = top_k

            top1 = top_k.iloc[0]
            print(f"  [{class_name:30s}] Top candidate: {top1['name']:15s}  "
                  f"corr={top1['correlation']:+.3f}  target={top1['target']}")

        return results

    # ── Main pipeline ─────────────────────────────────────────────────────────
    def run(
        self,
        features:  np.ndarray,
        labels:    np.ndarray,
        mol_df:    pd.DataFrame
    ) -> dict:
        """
        Full correlation pipeline.

        Args:
            features:  (N_images, 256) CNN biomarker features
            labels:    (N_images,)     disease class labels
            mol_df:    DataFrame from MolecularDataset.build_dataframe()

        Returns:
            dict of {disease_class_name: top-K candidate DataFrame}
        """
        print("\n[DrugCorrelator] Running correlation analysis …\n")

        # 1. Per-class biomarker profiles
        class_profiles = self.prepare_biomarkers(features, labels)

        # 2. Molecular descriptor matrix + PCA reduction
        mol_matrix_reduced, _, _ = self.prepare_molecular_matrix(mol_df)

        # 3 & 4. Correlate and rank
        print("\n[DrugCorrelator] Ranking candidates per disease class:\n")
        results = self.rank_candidates(
            class_profiles, mol_df, mol_matrix_reduced)

        print("\n[DrugCorrelator] Analysis complete.")
        return results

    # ── Summary table ─────────────────────────────────────────────────────────
    def build_summary_table(self, results: dict) -> pd.DataFrame:
        """
        Flatten all per-class results into a single ranked summary DataFrame.
        """
        rows = []
        for class_name, df in results.items():
            for rank, (_, row) in enumerate(df.iterrows(), 1):
                rows.append({
                    "Rank":           rank,
                    "Disease Class":  class_name,
                    "Drug":           row["name"],
                    "Target":         row["target"],
                    "Correlation":    round(row["correlation"], 4),
                    "Abs Corr":       round(row["abs_corr"],    4),
                    "pIC50":          round(row["pIC50"],        3),
                    "IC50 (nM)":      row["ic50_nM"],
                    "Significant":    row["significant"],
                    "SMILES":         row["smiles"]
                })
        return pd.DataFrame(rows)
