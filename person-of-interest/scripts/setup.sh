#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="/models/intel"
MODELS=(
  "face-detection-retail-0004"
  "landmarks-regression-retail-0009"
  "face-reidentification-retail-0095"
)
PRECISION="FP32"

echo "=== POI Setup Script ==="

# Create models directory
echo "[1/3] Creating models directory: $MODELS_DIR"
mkdir -p "$MODELS_DIR"

# Download models
echo "[2/3] Downloading OpenVINO models..."
if command -v omz_downloader &>/dev/null; then
  for model in "${MODELS[@]}"; do
    echo "  Downloading $model ($PRECISION)..."
    omz_downloader --name "$model" --precisions "$PRECISION" --output_dir "$MODELS_DIR" --cache_dir "$MODELS_DIR/.cache"
    echo "  ✓ $model downloaded"
  done
else
  echo "  WARNING: omz_downloader not found. Install with: pip install openvino-dev"
  echo "  Manual download commands:"
  for model in "${MODELS[@]}"; do
    echo "    omz_downloader --name $model --precisions $PRECISION --output_dir $MODELS_DIR"
  done
fi

# Check Redis
echo "[3/3] Checking Redis connectivity (port 6379)..."
if command -v redis-cli &>/dev/null; then
  if redis-cli -h localhost -p 6379 ping 2>/dev/null | grep -q PONG; then
    echo "  ✓ Redis is running"
  else
    echo "  ✗ Redis is NOT responding on port 6379"
  fi
elif command -v nc &>/dev/null; then
  if nc -z localhost 6379 2>/dev/null; then
    echo "  ✓ Redis port 6379 is open"
  else
    echo "  ✗ Redis port 6379 is not open"
  fi
else
  echo "  INFO: Neither redis-cli nor nc available to check Redis"
fi

echo ""
echo "=== Setup complete ==="
