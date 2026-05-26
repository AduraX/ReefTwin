# Contributing to ReefTwin

## Development Setup

```bash
# Clone and setup
git clone <repo-url> && cd ReefTwin
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,genai,mlops]"

# Install pre-commit hooks
pre-commit install

# Generate data and train models
make generate-sample-data && make ingest-noaa && make build-features
make train-model && make train-hybrid && make update-twin

# Run tests
make test
```

## Code Style

- Python: ruff (auto-enforced via pre-commit)
- Line length: 100
- Target: Python 3.11+
- Package manager: uv (not pip)
- Type hints encouraged on all public functions
- Docstrings on all public functions (Google style)

## Testing

- All new code must have tests
- Run `pytest -v` before submitting
- Target: maintain 158+ tests passing
- Integration tests in `tests/test_integration.py`

## Pull Request Process

1. Create a feature branch from `main`
2. Write code + tests
3. Run `make lint` and `make test`
4. Submit PR with description of changes
5. CI must pass (ruff + pytest + pipeline smoke test + security scan)

## Architecture

See the README for the architecture diagram.
The project follows a 7-tier enhancement plan with 35+ items implemented.
