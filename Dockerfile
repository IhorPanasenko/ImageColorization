# ── Stage 1: Build Vue frontend ───────────────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Install dependencies first (layer caching)
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

# Copy source and build
COPY frontend/ ./
RUN npm run build
# Produces /app/frontend/dist/

# ── Stage 2: Python runtime ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# System libraries required by OpenCV / Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install ML dependencies
COPY ml/requirements.txt ml/requirements.txt
RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r ml/requirements.txt

# Install API dependencies
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Copy project source
COPY ml/   ml/
COPY api/  api/
COPY run.py .

# Copy built Vue app from Stage 1
COPY --from=frontend-builder /app/frontend/dist frontend/dist

# Create runtime output directories
RUN mkdir -p outputs/checkpoints outputs/images outputs/runs outputs/uploads \
             data/test_samples

# Expose Flask (5000) and TensorBoard (6006)
EXPOSE 5000 6006

# Default: start the production web server
CMD ["python", "run.py"]
