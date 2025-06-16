# Fields of the World Inference API Server

This project provides a FastAPI-based implementation of the Fields of the World Inference API based on the OpenAPI specification. It enables running machine learning inference on satellite imagery using the `ftw-tools` package.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ftw-inference-api.git
   cd ftw-inference-api
   ```

2. Choose between one of the two methods:
   1. Install via conda:
      - Create a conda environment:
        ```
        conda env create -f server/env.yml
        ```
      - Activate the environment:
        ```bash
        conda activate ftw-inference-api
        ```

   2. Manually install the requirements:
      - Python 3.11 or 3.12
      - GDAL 3.11 or later with `libgdal-arrow-parquet`

1. Install dependencies:
   ```
   pip install -r server/requirements.txt
   ```

2. For development only: Install development dependencies:
   ```
   pip install -r server/requirements-dev.txt
   ```

3. For development only: Set up pre-commit hooks, see the [Code Quality](#code-quality) chapter.

## Running the Server

You can run the server using the provided `run.py` script in the server directory:

```bash
cd server
python run.py
```

### Command-line Options

- `--host HOST`: Specify the host address to bind the server to (default: 0.0.0.0)
- `--port PORT`: Specify the port to run the server on (default: 8000)
- `--config CONFIG`: Path to a custom config file
- `--debug`: Enable debug mode (enables auto-reload)

Example:

```bash
python run.py --host 127.0.0.1 --port 8080 --config /path/to/custom_config.yaml --debug
```

## Configuration

The server can be configured through a YAML file. By default, it looks for `config.yaml` in the `server/config` directory.

You can specify a custom configuration file using the `--config` command-line option:

```bash
python run.py --config /path/to/custom_config.yaml
```

## API Endpoints

The API provides the following endpoints:

- `GET /`: Root endpoint that returns basic API information
- `PUT /example`: Compute field boundaries for a small area quickly and return as GeoJSON
- `POST /projects`: Create a new project
- `GET /projects`: List all projects
- `GET /projects/{project_id}`: Get details of a specific project
- `DELETE /projects/{project_id}`: Delete a specific project
- `PUT /projects/{project_id}/images/{window}`: Upload an image for a project (window can be 'a' or 'b')
- `PUT /projects/{project_id}/inference`: Run inference on project images
- `PUT /projects/{project_id}/polygons`: Run polygonization on inference results
- `GET /projects/{project_id}/inference`: Get inference results for a project

## Authentication

The API uses Bearer token authentication. Include the `Authorization` header with a valid JWT token:

```http
Authorization: Bearer <your_token_here>
```

For development and testing, you can disable authentication by setting `auth_disabled` to `true` in `config/config.yaml`.

You still need to send a Bearer token to the API, but you can define a token via jwt.io for example.
The important part is that the secret key in config and in the config file align.
You also need to set the `sub` to `guest`.
For the default config, the following token can be used:
`eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJndWVzdCIsIm5hbWUiOiJHdWVzdCIsImlhdCI6MTc0ODIxNzYwMCwiZXhwaXJlcyI6OTk5OTk5OTk5OX0.lJIkuuSdE7ihufZwWtLx10D_93ygWUcUrtKhvlh6M8k`

## Development

### Project Structure

```
server/
│
├── app/                        # Main application package
│   ├── api/                    # API routes and endpoints
│   │   └── endpoints.py        # API endpoints implementation
│   │
│   ├── core/                   # Core application components
│   │   ├── auth.py             # Authentication utilities
│   │   ├── config.py           # Configuration management
│   │   └── inference.py        # Inference processing logic
│   │
│   ├── db/                     # Database related code
│   │   └── database.py         # Database session and engine setup
│   │
│   ├── models/                 # SQLAlchemy models
│   │   └── project.py          # Project and related models
│   │
│   ├── schemas/                # Pydantic schemas
│   │   └── project.py          # API request/response schemas
│   │
│   └── main.py                 # FastAPI application instance
│
├── config/                     # Configuration files
│   └── config.yaml             # Default configuration
│
├── test/                       # Test files
│   ├── conftest.py             # Pytest configuration and fixtures
│   └── test_api.py             # API endpoint tests
│
├── requirements.txt            # Project dependencies
└── run.py                      # Server startup script
```

### Code Quality

The project uses [Ruff](https://docs.astral.sh/ruff/) as the primary code quality tool, along with pre-commit hooks to ensure standards are met before committing changes. Ruff replaces multiple tools (black, isort, flake8) with a single, fast linter and formatter written in Rust.

The following checks are enabled:
- Code formatting with Ruff
- Code linting with Ruff (includes flake8-equivalent checks)
- Type checking with mypy
- Basic file checks (trailing whitespace, end of file, etc.)

To activate the pre-commit hooks, run:

```bash
pip install pre-commit
pre-commit install
```

You can manually run all pre-commit hooks on all files with:

```bash
pre-commit run --all-files
```

#### Using Ruff directly

Lint your code:
```bash
ruff check .
```

Format your code:
```bash
ruff format .
```

Auto-fix issues:
```bash
ruff check --fix .
```

### Running Tests

```bash
cd server
pytest -v
```

To run with coverage report:

```bash
pytest --cov=app
```

## CI/CD

This project includes GitHub Actions workflows in `.github/workflows/` for continuous integration:

- Running tests on multiple Python versions and operating systems
- Code linting and formatting with Ruff (via pre-commit)
- Coverage reporting

The CI pipeline uses pre-commit hooks to enforce code quality standards. Pre-commit automatically runs Ruff for linting and formatting, replacing the need for separate tools like black, isort, flake8, and mypy.

## License

See the [LICENSE](LICENSE) file for details.
