# AutoRec Backend

FastAPI backend for AutoRec.

## Requirements

- Python 3.11
- `pip`
- `docker` (optional, for container builds)

## Setup

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

## Environment

Copy `.env.example` to `.env` and adjust values as needed.

Key values:

- `DATABASE_URL` — SQLite by default
- `STORAGE_BACKEND` — `local`
- `UPLOAD_DIR` — local upload storage path

## Running locally

```bash
cd backend
.\.venv\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Tests

```bash
cd backend
.\.venv\Scripts\activate
pytest tests/ --cov=app --cov-report=term-missing
```

## Docker

Build and run the backend container:

```bash
cd backend
docker build -t autorec-backend .
docker run --rm -p 8000:8000 autorec-backend
```

The project includes `.dockerignore` to keep build context small and avoid bundling uploads or virtual environments.

## CI

Backend CI is configured in `.github/workflows/backend-ci.yml`.

The workflow installs dev dependencies, runs `ruff check app/`, executes tests with coverage, and uploads coverage to Codecov.

## Notes

- The app uses SQLite by default for local development.
- Uploaded dataset files and artifacts are stored under `uploads/`.
- Use `ENVIRONMENT=development` to enable docs and developer-friendly settings.
