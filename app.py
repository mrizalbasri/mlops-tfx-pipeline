"""Model Serving API with Prometheus Metrics."""

import glob
import os
import time
from typing import Any, Dict, List, Union

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field
import tensorflow as tf

# Prometheus Metrics Definitions
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP Requests",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP Request Latency in seconds",
    ["endpoint"],
)
PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Total Predictions Count",
    ["prediction_label"],
)
PREDICTION_VALUE_HIST = Histogram(
    "model_prediction_value",
    "Distribution of prediction probabilities",
    buckets=[0.1 * i for i in range(11)],
)
MODEL_LOADED = Gauge(
    "model_loaded_status",
    "Whether the serving model is loaded successfully (1=Yes, 0=No)",
)

# FastAPI Initialization
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Production ML Serving API powered by TFX SavedModel with Prometheus Monitoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
MODEL_INSTANCE = None
SERVING_SIGNATURE = None
MODEL_PATH = None

FLOAT_FEATURE_KEYS = {"oldpeak"}


def find_latest_model_path(base_dir: str = "serving_model_dir") -> str:
    """Finds the latest exported SavedModel directory."""
    dirs = sorted(glob.glob(os.path.join(base_dir, "*")))
    if not dirs:
        raise FileNotFoundError(f"No exported model found in {base_dir}")
    return dirs[-1]


def load_model():
    """Loads the SavedModel and extracts serving signature."""
    global MODEL_INSTANCE, SERVING_SIGNATURE, MODEL_PATH
    try:
        MODEL_PATH = find_latest_model_path()
        MODEL_INSTANCE = tf.saved_model.load(MODEL_PATH)
        SERVING_SIGNATURE = MODEL_INSTANCE.signatures["serving_default"]
        MODEL_LOADED.set(1)
        print(f"Loaded model successfully from: {MODEL_PATH}")
    except Exception as exc:
        MODEL_LOADED.set(0)
        print(f"Error loading model: {exc}")


@app.on_event("startup")
def startup_event():
    """Initializes model on startup."""
    load_model()


class PatientFeatures(BaseModel):
    """Schema for patient clinical features."""

    age: int = Field(..., example=63, description="Age in years")
    sex: int = Field(..., example=1, description="Sex (1 = male; 0 = female)")
    cp: int = Field(..., example=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., example=145, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., example=233, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., example=1, description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)")
    restecg: int = Field(..., example=0, description="Resting electrocardiographic results (0-2)")
    thalach: int = Field(..., example=150, description="Maximum heart rate achieved")
    exang: int = Field(..., example=0, description="Exercise induced angina (1 = yes; 0 = no)")
    oldpeak: float = Field(..., example=2.3, description="ST depression induced by exercise")
    slope: int = Field(..., example=0, description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(..., example=0, description="Number of major vessels (0-3)")
    thal: int = Field(..., example=1, description="Thalassemia (0=normal; 1=fixed; 2=reversible)")


class PredictionRequest(BaseModel):
    """Request payload containing one or multiple patient records."""

    inputs: List[PatientFeatures]


def serialize_example(data: Dict[str, Any]) -> bytes:
    """Serializes a feature dict into a tf.train.Example byte string matching schema."""
    example = tf.train.Example()
    for key, value in data.items():
        if key in FLOAT_FEATURE_KEYS:
            example.features.feature[key].float_list.value.append(float(value))
        else:
            example.features.feature[key].int64_list.value.append(int(value))
    return example.SerializeToString()


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Tracks latency and status code for each HTTP request."""
    start_time = time.time()
    endpoint = request.url.path
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        http_status=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)

    return response


@app.get("/")
def root():
    """Root metadata endpoint."""
    return {
        "service": "Heart Disease Prediction ML Serving API",
        "author": "M. Rizal Basri",
        "pipeline": "rizalbasri-pipeline",
        "model_path": MODEL_PATH,
        "status": "online",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "metrics": "/metrics",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": SERVING_SIGNATURE is not None,
    }


@app.post("/predict")
def predict(payload: Union[PredictionRequest, PatientFeatures]):
    """Predicts heart disease probability for given clinical features."""
    if SERVING_SIGNATURE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    if isinstance(payload, PatientFeatures):
        records = [payload.dict()]
    else:
        records = [item.dict() for item in payload.inputs]

    serialized_examples = [serialize_example(item) for item in records]

    try:
        raw_output = SERVING_SIGNATURE(examples=tf.constant(serialized_examples))
        output_tensor = list(raw_output.values())[0].numpy()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    results = []
    for prob_array in output_tensor:
        probability = float(prob_array[0])
        label = 1 if probability >= 0.5 else 0
        label_text = "Heart Disease Detected" if label == 1 else "Healthy / Normal"

        PREDICTIONS_TOTAL.labels(prediction_label=label_text).inc()
        PREDICTION_VALUE_HIST.observe(probability)

        results.append(
            {
                "probability": round(probability, 4),
                "prediction": label,
                "diagnosis": label_text,
            }
        )

    return {"predictions": results}


@app.get("/metrics")
def metrics():
    """Prometheus metrics scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
