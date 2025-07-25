#!/bin/bash
# Deployment script for Ubuntu Deep Learning AMI instances with Pixi and systemd service

set -e

# Parse command line arguments
BRANCH="main"
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--branch)
            BRANCH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-b|--branch BRANCH_NAME]"
            echo "  -b, --branch    Git branch to deploy (default: main)"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo "Deploying FTW Inference API on EC2..."
echo "Using branch: $BRANCH"

echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

echo "Installing system dependencies..."
sudo apt-get install -y \
    sqlite3 \
    libsqlite3-dev \
    curl

echo "Installing Pixi..."
if command -v pixi &> /dev/null; then
    echo "Pixi already installed, skipping..."
else
    curl -fsSL https://pixi.sh/install.sh | sh
    # Add pixi to PATH for current session
    export PATH="$HOME/.pixi/bin:$PATH"
    # Add to bashrc for future sessions
    echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
fi

echo "Cloning repository..."
cd ~
if [ -d "ftw-inference-api" ]; then
    echo "Repository already exists, switching to branch $BRANCH..."
    cd ftw-inference-api
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    git clone -b "$BRANCH" https://github.com/fieldsoftheworld/ftw-inference-api.git
    cd ftw-inference-api
fi

echo "Installing Pixi environment..."
# Ensure pixi is in PATH for this command
export PATH="$HOME/.pixi/bin:$PATH"
pixi install --environment production

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

# Create production environment file
echo "Creating production environment configuration..."
cat > /tmp/ftw-production.env << EOF
# Production Configuration
PROCESSING__GPU=0
CLOUDWATCH__ENABLED=true
S3__ENABLED=true
SECURITY__SECRET_KEY=$(openssl rand -hex 32)
SECURITY__AUTH_DISABLED=false
LOGGING__LEVEL=INFO
EOF

# Move environment file to proper location
sudo mkdir -p /etc/ftw-inference-api
sudo mv /tmp/ftw-production.env /etc/ftw-inference-api/production.env
sudo chown "$USER":"$USER" /etc/ftw-inference-api/production.env

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
WorkingDirectory=${APP_DIR}
Environment=PATH=${HOME_DIR}/.pixi/bin:\$PATH
EnvironmentFile=/etc/ftw-inference-api/production.env
ExecStart=${HOME_DIR}/.pixi/bin/pixi run --environment production python server/run.py --host 0.0.0.0 --port 8000
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
