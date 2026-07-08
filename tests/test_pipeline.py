"""
test_pipeline.py
────────────────
Unit tests for the full GenAI Medical Imaging pipeline.
Run with:  pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import torch
import yaml


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def dummy_images():
    """32 fake 64×64 RGB images in [-1, 1]."""
    return torch.randn(32, 3, 64, 64)

@pytest.fixture
def dummy_labels():
    return torch.randint(0, 9, (32,))

@pytest.fixture
def dummy_features():
    return np.random.randn(200, 256).astype(np.float32)

@pytest.fixture
def dummy_label_array():
    return np.random.randint(0, 9, 200)


# ── 1. Config Tests ───────────────────────────────────────────────────────────
class TestConfig:
    def test_config_loads(self, config):
        assert "gan" in config
        assert "cnn" in config
        assert "data" in config
        assert "molecular" in config

    def test_required_keys(self, config):
        assert config["data"]["num_classes"] == 9
        assert config["data"]["image_size"]  == 64
        assert config["gan"]["latent_dim"]   == 128
        assert config["cnn"]["feature_dim"]  == 256


# ── 2. Generator Tests ────────────────────────────────────────────────────────
class TestGenerator:
    def test_output_shape(self, config, device):
        from src.models.cgan import Generator
        G = Generator(
            latent_dim  = config["gan"]["latent_dim"],
            num_classes = config["data"]["num_classes"],
            img_size    = config["data"]["image_size"]
        ).to(device)

        z      = torch.randn(8, config["gan"]["latent_dim"]).to(device)
        labels = torch.randint(0, 9, (8,)).to(device)
        out    = G(z, labels)

        assert out.shape == (8, 3, 64, 64), f"Expected (8,3,64,64), got {out.shape}"

    def test_output_range(self, config, device):
        from src.models.cgan import Generator
        G = Generator(config["gan"]["latent_dim"], config["data"]["num_classes"],
                      img_size=config["data"]["image_size"]).to(device)
        z      = torch.randn(4, config["gan"]["latent_dim"]).to(device)
        labels = torch.randint(0, 9, (4,)).to(device)
        out    = G(z, labels)
        assert out.min() >= -1.01 and out.max() <= 1.01, "Generator output should be in [-1, 1]"

    def test_different_labels_different_output(self, config, device):
        from src.models.cgan import Generator
        G = Generator(config["gan"]["latent_dim"], config["data"]["num_classes"],
                      img_size=config["data"]["image_size"]).to(device)
        z = torch.randn(1, config["gan"]["latent_dim"]).to(device)
        out0 = G(z, torch.tensor([0]).to(device))
        out1 = G(z, torch.tensor([1]).to(device))
        assert not torch.allclose(out0, out1), "Same noise, different labels should produce different images"


# ── 3. Discriminator Tests ────────────────────────────────────────────────────
class TestDiscriminator:
    def test_output_shape(self, config, device, dummy_images, dummy_labels):
        from src.models.cgan import Discriminator
        D = Discriminator(config["data"]["num_classes"],
                          img_size=config["data"]["image_size"]).to(device)
        imgs   = dummy_images[:8].to(device)
        labels = dummy_labels[:8].to(device)
        out    = D(imgs, labels)
        assert out.shape == (8,), f"Expected (8,), got {out.shape}"

    def test_output_probability(self, config, device, dummy_images, dummy_labels):
        from src.models.cgan import Discriminator
        D = Discriminator(config["data"]["num_classes"],
                          img_size=config["data"]["image_size"]).to(device)
        out = D(dummy_images[:4].to(device), dummy_labels[:4].to(device))
        assert (out >= 0).all() and (out <= 1).all(), "Discriminator must output probabilities in [0,1]"


# ── 4. CNN Encoder Tests ──────────────────────────────────────────────────────
class TestCNNEncoder:
    def test_classifier_output(self, config, device, dummy_images):
        from src.models.cnn_encoder import CNNEncoder
        model = CNNEncoder(num_classes=9, feature_dim=256, pretrained=False).to(device)
        imgs  = dummy_images[:4].to(device)
        logits = model(imgs)
        assert logits.shape == (4, 9), f"Expected (4, 9), got {logits.shape}"

    def test_feature_output(self, config, device, dummy_images):
        from src.models.cnn_encoder import CNNEncoder
        model = CNNEncoder(num_classes=9, feature_dim=256, pretrained=False).to(device)
        imgs  = dummy_images[:4].to(device)
        logits, feats = model(imgs, return_features=True)
        assert feats.shape == (4, 256), f"Expected (4, 256), got {feats.shape}"
        assert logits.shape == (4, 9)

    def test_extract_features(self, config, device, dummy_images):
        from src.models.cnn_encoder import CNNEncoder
        model = CNNEncoder(num_classes=9, feature_dim=256, pretrained=False).to(device)
        feats = model.extract_features(dummy_images[:8].to(device))
        assert feats.shape == (8, 256)


# ── 5. Drug Correlator Tests ──────────────────────────────────────────────────
class TestDrugCorrelator:
    def test_biomarker_profiles(self, config, dummy_features, dummy_label_array):
        from src.models.drug_correlator import DrugCorrelator
        dc = DrugCorrelator(config)
        profiles = dc.prepare_biomarkers(dummy_features, dummy_label_array)
        assert len(profiles) > 0
        for cls, vec in profiles.items():
            assert vec.shape == (256,)

    def test_molecular_matrix(self, config):
        from src.models.drug_correlator import DrugCorrelator
        from src.data.dataset import MolecularDataset
        dc     = DrugCorrelator(config)
        mol_ds = MolecularDataset()
        mol_df = mol_ds.build_dataframe()
        mat, _, _ = dc.prepare_molecular_matrix(mol_df)
        assert mat.ndim == 2
        assert mat.shape[0] == len(mol_df)

    def test_full_run(self, config, dummy_features, dummy_label_array):
        from src.models.drug_correlator import DrugCorrelator
        from src.data.dataset import MolecularDataset
        dc      = DrugCorrelator(config)
        mol_ds  = MolecularDataset()
        mol_df  = mol_ds.build_dataframe()
        results = dc.run(dummy_features, dummy_label_array, mol_df)
        assert isinstance(results, dict)
        assert len(results) > 0
        for class_name, df in results.items():
            assert "name" in df.columns
            assert "correlation" in df.columns
            assert len(df) <= config["molecular"]["top_k_candidates"]


# ── 6. Metrics Tests ──────────────────────────────────────────────────────────
class TestMetrics:
    def test_classification_metrics(self):
        from src.utils.metrics import compute_classification_metrics
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 1, 0, 2])
        res = compute_classification_metrics(y_true, y_pred)
        assert "accuracy"    in res
        assert "f1_macro"    in res
        assert 0.0 <= res["accuracy"] <= 1.0

    def test_correlation_summary(self, config, dummy_features, dummy_label_array):
        from src.models.drug_correlator import DrugCorrelator
        from src.data.dataset import MolecularDataset
        from src.utils.metrics import compute_correlation_summary
        dc      = DrugCorrelator(config)
        mol_df  = MolecularDataset().build_dataframe()
        results = dc.run(dummy_features, dummy_label_array, mol_df)
        summary = compute_correlation_summary(results)
        assert "_overall" in summary
        assert summary["_overall"]["mean_abs_corr"] >= 0

    def test_gan_health_check(self):
        from src.utils.metrics import check_gan_health
        history = {
            "g_loss": [2.0, 1.8, 1.5, 1.3, 1.2, 1.1, 1.05, 1.02, 1.01, 1.00],
            "d_loss": [0.6, 0.55, 0.52, 0.51, 0.50, 0.50, 0.49, 0.49, 0.50, 0.50]
        }
        health = check_gan_health(history)
        assert "mode_collapse_risk" in health
        assert "converged"          in health
        assert health["mode_collapse_risk"] in ("LOW", "MEDIUM", "HIGH")


# ── 7. Molecular Dataset Tests ────────────────────────────────────────────────
class TestMolecularDataset:
    def test_build_dataframe(self):
        from src.data.dataset import MolecularDataset
        mol_ds = MolecularDataset()
        df     = mol_ds.build_dataframe()
        assert len(df) > 0
        assert "name"        in df.columns
        assert "smiles"      in df.columns
        assert "pIC50"       in df.columns
        assert "descriptors" in df.columns

    def test_descriptor_shape(self):
        from src.data.dataset import MolecularDataset
        mol_ds = MolecularDataset(descriptor_type="morgan", morgan_bits=2048)
        df     = mol_ds.build_dataframe()
        for _, row in df.iterrows():
            assert row["descriptors"].shape == (2048,)

    def test_invalid_smiles_skipped(self):
        from src.data.dataset import MolecularDataset
        mol_ds = MolecularDataset()
        desc   = mol_ds.compute_descriptors("INVALID_SMILES_XYZ")
        assert desc is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
