FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN uv pip install --system --no-cache \
    "tensorflow-cpu==2.13.1" \
    "fastapi" \
    "uvicorn" \
    "prometheus-client" \
    "pydantic"

COPY modules/ modules/
COPY serving_model_dir/ serving_model_dir/
COPY app.py .
COPY sample_request.json .
COPY requirements.txt .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
