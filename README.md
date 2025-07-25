# Fields of the World Inference API Server

This project provides a FastAPI-based implementation of the Fields of the World Inference API based on the OpenAPI specification. It enables running machine learning inference on satellite imagery using the `ftw-tools` package.

## Installation

1. Install [Pixi](https://pixi.sh/):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | sh  # macOS/Linux
   # or: brew install pixi
   ```

2. Clone and setup:
   ```bash
   git clone https://github.com/fieldsoftheworld/ftw-inference-api.git
   cd ftw-inference-api
   pixi install
   ```

## Deployment

### Quick EC2 Deployment

For rapid deployment on AWS EC2 instances using Ubuntu Deep Learning AMI with NVIDIA drivers:

```bash
curl -L https://raw.githubusercontent.com/fieldsoftheworld/ftw-inference-api/main/deploy.sh | bash
```

To deploy a specific branch:
```bash
curl -L https://raw.githubusercontent.com/fieldsoftheworld/ftw-inference-api/main/deploy.sh | bash -s -- -b your-branch-name
```

This script will:
- Install Pixi package manager
- Clone the repository and checkout the specified branch
- Install dependencies using Pixi production environment
- Download all pre-trained model checkpoints (~800MB total)
- Enable GPU support in configuration
- Configure a systemd service for automatic startup
- Set up log rotation

**Service management:**
```bash
sudo systemctl status ftw-inference-api     # Check status
sudo systemctl start ftw-inference-api      # Start service
sudo systemctl stop ftw-inference-api       # Stop service
sudo systemctl restart ftw-inference-api    # Restart service
sudo journalctl -u ftw-inference-api -f     # Follow logs
sudo journalctl -u ftw-inference-api --since today  # Today's logs
```

## Running the Server in Development Mode

```bash
pixi run start  # Development server with debug mode and auto reload
```

Or run directly with options:
```bash
pixi run python server/run.py --host 127.0.0.1 --port 8080 --debug
```

**Command-line options:**
- `--host HOST`: Host address (default: 0.0.0.0)
- `--port PORT`: Port number (default: 8000)
- `--config CONFIG`: Custom config file path
- `--debug`: Enable debug mode and auto-reload

## Configuration

The server loads configuration from `server/config/base.toml` by default. Settings can be overridden using environment variables with double underscore delimiter (e.g., `SECURITY__SECRET_KEY`).

You can specify a custom configuration file using the `--config` command-line option:

```bash
python run.py --config /path/to/custom_config.toml
```

## API Endpoints

The API provides the following versioned endpoints under `/v1/`:

- `GET /`: Root endpoint that returns basic API information
- `PUT /v1/example`: Compute field boundaries for a small area quickly and return as GeoJSON
- `POST /v1/projects`: Create a new project
- `GET /v1/projects`: List all projects
- `GET /v1/projects/{project_id}`: Get details of a specific project
- `DELETE /v1/projects/{project_id}`: Delete a specific project
- `PUT /v1/projects/{project_id}/images/{window}`: Upload an image for a project (window can be 'a' or 'b')
- `PUT /v1/projects/{project_id}/inference`: Run inference on project images
- `PUT /v1/projects/{project_id}/polygons`: Run polygonization on inference results
- `GET /v1/projects/{project_id}/inference`: Get inference results for a project

## Authentication

The API uses Bearer token authentication. Include the `Authorization` header with a valid JWT token:

```http
Authorization: Bearer <your_token_here>
```

For development and testing, you can disable authentication by setting `auth_disabled` to `true` in `server/config/base.toml`.

You still need to send a Bearer token to the API, but you can define a token via jwt.io for example.
The important part is that the secret key in config and in the config file align.
You also need to set the `sub` to `guest`.
For the default config, the following token can be used:
`eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJndWVzdCIsIm5hbWUiOiJHdWVzdCIsImlhdCI6MTc0ODIxNzYwMCwiZXhwaXJlcyI6OTk5OTk5OTk5OX0.lJIkuuSdE7ihufZwWtLx10D_93ygWUcUrtKhvlh6M8k`

## Development

### Project Structure

The application follows clean architecture principles with clear separation of concerns:

```
server/
├── app/                        # Main application package
│   ├── api/v1/                 # API endpoints and dependencies
│   ├── services/               # Business logic layer
│   ├── ml/                     # ML pipeline and validation
│   ├── core/                   # Infrastructure (auth, config, storage)
│   ├── schemas/                # Pydantic request/response models
│   ├── models/                 # Database models
│   ├── db/                     # Database connection and utilities
│   └── main.py                 # FastAPI application setup
├── config/                     # Configuration files
├── data/                       # ML models, results, temp files
├── tests/                      # Test suite
└── run.py                      # Development server runner
```

### Code Quality

Uses [Ruff](https://docs.astral.sh/ruff/) for linting/formatting and pre-commit hooks for quality checks.

```bash
pixi run lint     # Run all pre-commit hooks
pixi run format   # Format code
pixi run check    # Check without fixing
```

Setup pre-commit:
```bash
pixi run pre-commit install
```

### Running Tests

```bash
pixi run test  # All tests with coverage
```

## License

See the [LICENSE](LICENSE) file for details.
