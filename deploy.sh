#!/bin/bash
# Deployment script for Ubuntu Deep Learning AMI instances with UV and systemd service

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

echo "Installing UV..."
if command -v uv &> /dev/null; then
    echo "UV already installed, skipping..."
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    # Add to bashrc for future sessions
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

echo "Installing Python 3.12 via UV..."
uv python install 3.12

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

echo "Installing Python dependencies with UV..."
# Ensure uv is in PATH for this command
export PATH="$HOME/.local/bin:$PATH"
uv sync

echo "Creating data directories..."
mkdir -p server/data/uploads server/data/results server/logs

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

# Build the ExecStart command using UV
BASE_EXEC_START="${HOME_DIR}/.local/bin/uv run python run.py --host 0.0.0.0 --port 8000"

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
Environment="PATH=${HOME_DIR}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
EnvironmentFile=/etc/ftw-inference-api/production.env
ExecStart=${BASE_EXEC_START}
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
