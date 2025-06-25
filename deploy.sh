#!/bin/bash
# Deployment script for Ubuntu Deep Learning AMI instances with miniforge and systemd service

set -e

echo "Deploying FTW Inference API on EC2..."

echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

echo "Installing system dependencies..."
sudo apt-get install -y \
    sqlite3 \
    libsqlite3-dev

echo "Installing miniforge..."
if [ -d "$HOME/miniforge3" ]; then
    echo "Miniforge already installed, skipping..."
else
    cd /tmp
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    chmod +x Miniforge3-Linux-x86_64.sh
    ./Miniforge3-Linux-x86_64.sh -b -p "$HOME"/miniforge3

    # Initialize conda
    "$HOME"/miniforge3/bin/conda init bash
fi

# Setup PATH
# shellcheck disable=SC2016
echo 'export PATH="$HOME/miniforge3/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/miniforge3/bin:$PATH"

# shellcheck disable=SC1090
source ~/.bashrc

echo "Cloning repository..."
cd ~
if [ -d "ftw-inference-api" ]; then
    echo "Repository already exists, pulling latest changes..."
    cd ftw-inference-api
    git pull
else
    git clone https://github.com/fieldsoftheworld/ftw-inference-api.git
    cd ftw-inference-api
fi

echo "Creating conda environment..."
conda env create -f server/env.yml -y

echo "Downloading FTW model checkpoints..."
mkdir -p server/data/uploads server/data/results server/data/models server/logs
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
        echo "$model already exists"
    else
        echo "Downloading $model..."
        if wget -q --show-progress "$BASE_URL$model" -O "$MODELS_DIR/$model"; then
            echo "$model downloaded"
        else
            echo "Failed to download $model"
        fi
    fi
done

# Enable GPU in config (replace gpu: null with gpu: 0)
sed -i 's/gpu: null/gpu: 0/' server/config/config.yaml

# Enable CloudWatch logging
sed -i 's/enabled: false/enabled: true/' server/config/config.yaml

chmod +x server/run.py

echo "Creating systemd service..."
APP_DIR="$(pwd)"
USER="$(whoami)"
HOME_DIR="$(eval echo ~$USER)"

sudo tee /etc/systemd/system/ftw-inference-api.service > /dev/null <<EOF
[Unit]
Description=FTW Inference API
After=network.target

[Service]
Type=simple
User=${USER}
Group=${USER}
WorkingDirectory=${APP_DIR}/server
ExecStart=/bin/bash -c "cd ${APP_DIR}/server && source ${HOME_DIR}/miniforge3/bin/activate ftw-inference-api && python run.py --host 0.0.0.0 --port 8000"
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ftw-inference-api

# Security settings
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
EOF

# Setup log rotation
echo "Setting up log rotation..."
sudo tee /etc/logrotate.d/ftw-inference-api > /dev/null <<EOF
/var/log/ftw-inference-api/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
    postrotate
        systemctl reload ftw-inference-api || true
    endscript
}
EOF

# Create log directory
sudo mkdir -p /var/log/ftw-inference-api
sudo chown "$USER":"$USER" /var/log/ftw-inference-api

echo "Enabling and starting the service..."
sudo systemctl daemon-reload
sudo systemctl enable ftw-inference-api
sudo systemctl start ftw-inference-api

# Wait a moment for the service to start
sleep 5

echo "Checking service status..."
sudo systemctl status ftw-inference-api --no-pager

echo "Deployment complete!"
echo ""
echo "Configuration:"
echo "   GPU enabled (gpu: 0)"
echo ""
echo "Service Management:"
echo "  sudo systemctl status ftw-inference-api     # Check status"
echo "  sudo systemctl start ftw-inference-api      # Start service"
echo "  sudo systemctl stop ftw-inference-api       # Stop service"
echo "  sudo systemctl restart ftw-inference-api    # Restart service"
echo ""
echo "Logs:"
echo "  sudo journalctl -u ftw-inference-api -f     # Follow logs"
echo "  sudo journalctl -u ftw-inference-api --since today"
