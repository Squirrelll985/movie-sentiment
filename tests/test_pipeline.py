# -*- coding: utf-8 -*-
"""Unit tests for preprocessing, model, metrics and the Flask API."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment.config import ModelConfig  # noqa: E402
from sentiment.data import clean_text, tokenize  # noqa: E402
from sentiment.metrics import compute_metrics, roc_auc_score  # noqa: E402
from sentiment.model import build_model, count_parameters, save_model_checkpoint  # noqa: E402


class TestPreprocessing(unittest.TestCase):
    def test_clean_html_and_case(self):
        self.assertEqual(clean_text("A <b>Great</b> Film<br />Really!"), "a great film really!")

    def test_tokenizer_keeps_negation_and_punctuation(self):
        tokens = tokenize("I don't like it. Not good!")
        self.assertIn("don't", tokens)
        self.assertIn("!", tokens)
        self.assertNotIn(".", tokens)

    def test_tokenizer_handles_unicode_quotes(self):
        self.assertEqual(tokenize("It’s fine"), ["it's", "fine"])


class TestMetrics(unittest.TestCase):
    def test_perfect_classifier(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = compute_metrics(y, p)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_auc_is_symmetric(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.2, 0.7, 0.6, 0.8])
        auc = roc_auc_score(y, p)
        self.assertAlmostEqual(auc, roc_auc_score(1 - y, 1 - p), places=6)


class TestModel(unittest.TestCase):
    def test_bilstm_attention_forward(self):
        config = ModelConfig(embedding_dim=16, hidden_dim=16, dropout=0.0)
        model = build_model("bilstm_attention", 100, config)
        ids = torch.randint(1, 99, (4, 20))
        lengths = torch.tensor([20, 15, 10, 3])
        output = model(ids, lengths)
        self.assertEqual(output["logits"].shape, (4, 2))
        self.assertEqual(output["attention"].shape, (4, 20))
        self.assertTrue(torch.allclose(output["attention"].sum(dim=1), torch.ones(4), atol=1e-5))
        self.assertGreater(count_parameters(model), 0)

    def test_textcnn_forward(self):
        config = ModelConfig(embedding_dim=16, num_filters=8, dropout=0.0)
        model = build_model("textcnn", 100, config)
        output = model(torch.randint(1, 99, (3, 20)))
        self.assertEqual(output["logits"].shape, (3, 2))

    def test_padding_rows_are_masked(self):
        config = ModelConfig(embedding_dim=8, hidden_dim=8, dropout=0.0)
        model = build_model("bilstm_attention", 50, config).eval()
        ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 4, 5]])
        lengths = torch.tensor([3, 5])
        with torch.no_grad():
            attention = model(ids, lengths)["attention"]
        self.assertEqual(float(attention[0, 3]), 0.0)
        self.assertEqual(float(attention[0, 4]), 0.0)


class TestFlaskAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        vocab = {"<pad>": 0, "<unk>": 1}
        for i, word in enumerate(["good", "bad", "film", "great", "boring", "story"], start=2):
            vocab[word] = i
        config = ModelConfig(embedding_dim=8, hidden_dim=8, dropout=0.0)
        model = build_model("bilstm_attention", len(vocab), config)
        checkpoint = tmp / "model.pt"
        save_model_checkpoint(
            model,
            checkpoint,
            "bilstm_attention",
            config,
            vocab,
            {"accuracy": 0.9},
            extra={"data_config": {"max_len": 16}},
        )
        examples = tmp / "examples.json"
        examples.write_text(
            json.dumps([{"label": "positive", "text": "a great film"}]), encoding="utf-8"
        )
        from app import create_app

        cls.client = create_app(checkpoint, tmp / "metrics.json", examples).test_client()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_predict(self):
        response = self.client.post("/api/predict", json={"text": "a great film", "explain": True})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn(payload["label"], {"positive", "negative"})
        self.assertGreaterEqual(payload["confidence"], 0.5)
        self.assertIn("explanation", payload)

    def test_empty_text_rejected(self):
        response = self.client.post("/api/predict", json={"text": "   "})
        self.assertEqual(response.status_code, 400)

    def test_batch(self):
        response = self.client.post("/api/batch", json={"texts": ["good film", "bad film"]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["count"], 2)


if __name__ == "__main__":
    unittest.main()
