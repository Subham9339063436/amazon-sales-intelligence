# ---------------------------------------------------------------------------
# Amazon Sales Forecasting — API + dashboard image
#
# One image, two processes (see docker-compose.yml):
#   API       : gunicorn -b 0.0.0.0:5001 api.app:app   (default CMD)
#   Dashboard : streamlit run app.py                    (command override)
#
# Build:  docker build -t amazon-sales-forecasting .
# No secrets are baked into the image; no credentials are required to run it.
# ---------------------------------------------------------------------------
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp1 — OpenMP runtime required by LightGBM's native library.
# curl       — convenient for `docker exec ... curl localhost:5001/health`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch first: serving runs on CPU, and this avoids pulling the
# multi-GB CUDA wheels from PyPI (requirements.txt then sees torch satisfied).
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"

COPY requirements.txt .
RUN pip install -r requirements.txt

# Runtime files. data/ (CSVs) and results/ + models/ (artifacts) are required
# by the dashboard and the API; scripts/ and tests/ are included so the image
# can reproduce training and run the suite if desired.
COPY src/ src/
COPY api/ api/
COPY dashboard/ dashboard/
COPY scripts/ scripts/
COPY tests/ tests/
COPY data/ data/
COPY results/ results/
COPY models/ models/
COPY app.py .
COPY README.md .

EXPOSE 5001 8501

# Default: REST API. Start the dashboard by overriding the command:
#   docker run --rm -p 8501:8501 amazon-sales-forecasting \
#     streamlit run app.py --server.address 0.0.0.0 --server.headless true
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--timeout", "120", "api.app:app"]
