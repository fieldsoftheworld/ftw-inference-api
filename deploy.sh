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
sudo apt-get install -y  curl

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
pixi install

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

# Create minimal production environment file (optional customizations)
echo "Creating minimal production environment configuration..."
cat > /tmp/ftw-production.env << EOF
# Production Configuration - Add any environment variable overrides here
# The application will use base.toml defaults unless overridden
PROCESSING__GPU=0
SECURITY__AUTH_DISABLED=true
LOGGING__LEVEL=INFO
STORAGE__SOURCE_COOP__USE_STS_WORKAROUND=true
EOF

# Move environment file to proper location
sudo mkdir -p /etc/ftw-inference-api
sudo mv /tmp/ftw-production.env /etc/ftw-inference-api/production.env
sudo chown "$USER":"$USER" /etc/ftw-inference-api/production.env

echo "Environment file created at /etc/ftw-inference-api/production.env"
echo "Modify this file to override any settings from base.toml as needed"

chmod +x server/run.py

echo "Creating systemd service..."
APP_DIR="$(pwd)"
USER="$(whoami)"
HOME_DIR="$(eval echo ~$USER)"

# Conditionally build the ExecStart command based on the GPU configuration.
# We check the final destination of the production.env file.
BASE_EXEC_START="${HOME_DIR}/.pixi/bin/pixi run --environment production python run.py --host 0.0.0.0 --port 8000"
EXEC_START_CMD=$BASE_EXEC_START

if grep -q "PROCESSING__GPU=null" /etc/ftw-inference-api/production.env; then
    echo "GPU is disabled. Applying CONDA_OVERRIDE_GLIBC to the service."
    # Prepend the environment variable directly to the command.
    EXEC_START_CMD="CONDA_OVERRIDE_GLIBC=2.17 ${BASE_EXEC_START}"
fi

sudo tee /etc/systemd/system/ftw-inference-api.service > /dev/null <<EOF
[Unit]
Description=FTW Inference API
After=network.target

[Service]
Type=simple
User=${USER}
Group=${USER}
WorkingDirectory=${APP_DIR}/server
# Set a reliable PATH for the service environment.
Environment="PATH=${HOME_DIR}/.pixi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
EnvironmentFile=/etc/ftw-inference-api/production.env
# Use /bin/sh -c to correctly process the command string with the conditional variable.
ExecStart=/bin/sh -c "${EXEC_START_CMD}"
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
sudo systemctl restart ftw-inference-api # Use restart to ensure changes are applied

# Wait a moment for the service to start
sleep 5

echo "Checking service status..."
sudo systemctl status ftw-inference-api --no-pager

echo "Deployment complete!"
echo ""
echo "Configuration:"
echo "   Check /etc/ftw-inference-api/production.env for current settings."
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
