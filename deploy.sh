#!/bin/bash
# Simple deployment script for EC2 Ubuntu instances

set -e

echo "Deploying FTW Inference API on EC2..."

echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

echo "Installing system dependencies..."
sudo apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libgdal-arrow-parquet-dev \
    python3-gdal \
    sqlite3 \
    libsqlite3-dev \
    wget \
    curl

# Install UV (fast Python package manager)
echo "Installing UV..."
curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh | sh
source ~/.bashrc
export PATH="$HOME/.cargo/bin:$PATH"

echo "Installing Python dependencies with UV..."
uv sync

echo "Creating data directories..."
mkdir -p server/data/uploads server/data/results server/data/models

# Download model checkpoints
echo "Downloading FTW model checkpoints..."
BASE_URL="https://github.com/fieldsoftheworld/ftw-baselines/releases/download/v1/"
MODELS_DIR="server/data/models"

MODELS=(
    "2_Class_FULL_FTW_Pretrained.ckpt"
    "2_Class_CCBY_FTW_Pretrained.ckpt"
    "3_Class_FULL_FTW_Pretrained.ckpt"
    "3_Class_CCBY_FTW_Pretrained.ckpt"
)

for model in "${MODELS[@]}"; do
    if [ -f "$MODELS_DIR/$model" ]; then
        echo "✅ $model already exists"
    else
        echo "Downloading $model..."
        if wget -q --show-progress "$BASE_URL$model" -O "$MODELS_DIR/$model"; then
            echo "✅ $model downloaded"
        else
            echo "❌ Failed to download $model"
        fi
    fi
done

chmod +x run.py

echo "Deployment complete!"
echo ""
echo "To start the server:"
echo "  ./run.py"
echo ""
echo "For development with auto-reload:"
echo "  ./run.py --debug"