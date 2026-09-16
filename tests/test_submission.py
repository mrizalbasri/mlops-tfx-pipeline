"""Submission completeness and sanity test suite."""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient
from app import app, load_model


def test_required_submission_files_exist():
    required_files = [
        "README.md",
        "requirements.txt",
        "Dockerfile",
        "docker-compose.yml",
        "pipeline.py",
        "app.py",
        "sample_request.json",
        "rizalbasri-pipeline.ipynb",
        "rizalbasri-testing.ipynb",
        "rizalbasri-pylint.png",
        "rizalbasri-deployment.png",
        "rizalbasri-monitoring.png",
        "rizalbasri-grafana-dashboard.png",
        "data/heart.csv",
        "modules/__init__.py",
        "modules/transform.py",
        "modules/trainer.py",
        "modules/components.py",
        "monitoring/Dockerfile",
        "monitoring/prometheus.yml",
        "monitoring/prometheus.config",
    ]
    for rel_path in required_files:
        assert os.path.exists(rel_path), f"Missing required file: {rel_path}"


def test_notebooks_are_valid_json():
    notebooks = ["rizalbasri-pipeline.ipynb", "rizalbasri-testing.ipynb", "notebook.ipynb"]
    for nb in notebooks:
        with open(nb, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "cells" in data
            assert len(data["cells"]) > 0


def test_app_endpoints():
    load_model()
    client = TestClient(app)

    # Test Root
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["service"] == "Heart Disease Prediction ML Serving API"

    # Test Health
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"
    assert res_health.json()["model_loaded"] is True

    # Test Predict
    with open("sample_request.json") as f:
        payload = json.load(f)
    res_pred = client.post("/predict", json=payload)
    assert res_pred.status_code == 200
    preds = res_pred.json()["predictions"]
    assert len(preds) == 2
    assert "probability" in preds[0]
    assert "diagnosis" in preds[0]

    # Test Metrics
    res_metrics = client.get("/metrics")
    assert res_metrics.status_code == 200
    assert "http_requests_total" in res_metrics.text
    assert "model_predictions_total" in res_metrics.text
