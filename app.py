# -*- coding: utf-8 -*-
"""Flask web application for the LSTM sentiment analysis system."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment.predict import SentimentPredictor  # noqa: E402

DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "main" / "best_model.pt"
DEFAULT_METRICS = PROJECT_ROOT / "outputs" / "main" / "metrics.json"
DEFAULT_EXAMPLES = PROJECT_ROOT / "assets" / "examples.json"


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def create_app(
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    examples_path: str | Path | None = None,
) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    checkpoint = Path(checkpoint_path or os.environ.get("SENTIMENT_CHECKPOINT", DEFAULT_CHECKPOINT))
    metrics_file = Path(metrics_path or os.environ.get("SENTIMENT_METRICS", DEFAULT_METRICS))
    examples_file = Path(examples_path or DEFAULT_EXAMPLES)

    state: dict[str, Any] = {"predictor": None, "error": None}
    lock = threading.Lock()

    def get_predictor() -> SentimentPredictor:
        if state["predictor"] is None:
            with lock:
                if state["predictor"] is None:
                    try:
                        state["predictor"] = SentimentPredictor(checkpoint, device="cpu")
                        state["error"] = None
                    except Exception as exc:  # keep the UI alive with a clear message
                        state["error"] = str(exc)
                        raise
        return state["predictor"]

    @app.get("/")
    def index():
        return render_template("index.html", model_ready=checkpoint.exists())

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok" if checkpoint.exists() else "model_missing",
                "checkpoint": str(checkpoint),
                "loaded": state["predictor"] is not None,
                "error": state["error"],
            }
        )

    @app.get("/api/examples")
    def examples():
        return jsonify(load_json(examples_file, []))

    @app.get("/api/model")
    def model_info():
        info = {
            "checkpoint": checkpoint.name,
            "exists": checkpoint.exists(),
            "metrics": {},
            "parameter_count": None,
            "vocab_size": None,
            "max_len": None,
            "model_name": None,
        }
        metrics = load_json(metrics_file, {})
        if metrics:
            info["metrics"] = metrics.get("test_metrics", {})
            info["parameter_count"] = metrics.get("parameter_count")
            info["model_name"] = metrics.get("model")
        if checkpoint.exists():
            try:
                predictor = get_predictor()
                info.update(
                    {
                        "parameter_count": predictor.parameter_count,
                        "vocab_size": len(predictor.vocab),
                        "max_len": predictor.max_len,
                        "model_name": predictor.model_name,
                    }
                )
            except Exception as exc:
                info["error"] = str(exc)
        return jsonify(info)

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("text", "")).strip()
        if not text:
            return jsonify({"error": "请输入影评文本。"}), 400
        if len(text) > 10000:
            return jsonify({"error": "文本过长，请控制在 10000 个字符以内。"}), 400
        if not checkpoint.exists():
            return jsonify({"error": "模型尚未训练，请先运行训练脚本生成 best_model.pt。"}), 503
        try:
            predictor = get_predictor()
            result = predictor.predict(
                text,
                explain=bool(payload.get("explain", True)),
                top_k=int(payload.get("top_k", 6)),
            )
            return jsonify(result)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"推理失败：{exc}"}), 500

    @app.post("/api/batch")
    def batch():
        payload = request.get_json(silent=True) or {}
        texts = payload.get("texts", [])
        if not isinstance(texts, list) or not texts:
            return jsonify({"error": "texts 必须是非空数组。"}), 400
        if len(texts) > 50:
            return jsonify({"error": "单次最多分析 50 条文本。"}), 400
        if not checkpoint.exists():
            return jsonify({"error": "模型尚未训练。"}), 503
        try:
            predictor = get_predictor()
            results = [predictor.predict(str(text), explain=False) for text in texts]
            return jsonify({"count": len(results), "results": results})
        except Exception as exc:
            return jsonify({"error": f"批量推理失败：{exc}"}), 500

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"error": "请求体过大。"}), 413

    return app


app = create_app()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sentiment analysis web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    application = create_app(args.checkpoint) if args.checkpoint else app
    application.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
