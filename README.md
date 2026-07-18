# AutoRec

AutoRec is a full-stack machine learning recommendation platform that automates model selection, training, evaluation, and explainability. It features a Python/FastAPI backend with comprehensive ML engines and a modern React/Vite frontend for intuitive experiment management.

## Features

- **Automated ML Pipeline**: Upload datasets, analyze data characteristics, and get intelligent preprocessing recommendations
- **Multi-Algorithm Training**: Train 20+ classification and regression models automatically
- **Model Evaluation**: Comprehensive metrics with cross-validation and overfitting detection
- **Intelligent Recommendations**: AI-powered model selection based on performance, interpretability, and training time
- **Explainability**: Feature importance and SHAP values for model transparency
- **Experiment Tracking**: Full lifecycle management of ML experiments
- **Report Generation**: Export results in multiple formats

## Table of Contents

- [Installation](#installation)
- [Local Development](#local-development)
- [Running with Docker](#running-with-docker)
- [CI/CD](#cicd)
- [API Documentation](#api-documentation)
- [Project Architecture](#project-architecture)
- [Folder Structure](#folder-structure)
- [Supported Algorithms](#supported-algorithms)
- [Screenshots](#screenshots)
- [Future Roadmap](#future-roadmap)
- [Contributing](#contributing)

## Installation

### Prerequisites

- **Backend**: Python 3.11+, pip
- **Frontend**: Node.js 20+, npm
- **Docker**: Docker Desktop (optional)

### Backend Setup

```bash
cd backend
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Unix:
source .venv/bin/activate
pip install -e ".[dev]"
```

### Frontend Setup

```bash
cd frontend
npm install
```

## Local Development

### Starting the Backend

```bash
cd backend
.\.venv\Scripts\activate  # Windows
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`
API docs: `http://localhost:8000/docs`

### Starting the Frontend

```bash
cd frontend
npm run dev
```

The UI will be available at `http://localhost:5173`

### Running Tests

```bash
# Backend tests
cd backend
pytest tests/ -v --cov=app --cov-report=term-missing

# Frontend tests
cd frontend
npm run test:coverage
```

## Running with Docker

### Using Docker Compose (Recommended)

```bash
docker-compose up --build
```

This will start both backend (port 8000) and frontend (port 80) services.

### Individual Containers

```bash
# Backend
cd backend
docker build -t autorec-backend .
docker run --rm -p 8000:8000 -v $(pwd)/uploads:/app/uploads autorec-backend

# Frontend
cd frontend
docker build -t autorec-frontend .
docker run --rm -p 80:80 autorec-frontend
```

### Environment Variables

Backend environment variables (see `backend/.env.example`):

- `ENVIRONMENT` - `development` or `production`
- `DATABASE_URL` - Database connection string
- `STORAGE_BACKEND` - `local` or cloud storage
- `UPLOAD_DIR` - Path for uploaded files
- `MAX_UPLOAD_SIZE_MB` - Maximum file upload size
- `CORS_ORIGINS` - Allowed CORS origins

## CI/CD

This project uses GitHub Actions for continuous integration and deployment. The workflows are located in `.github/workflows/`.

### Workflows

- **Backend CI** (`.github/workflows/backend-ci.yml`): Runs tests, linting, and builds Docker images for the backend
- **Frontend CI** (`.github/workflows/frontend-ci.yml`): Runs tests, linting, type checking, and builds Docker images for the frontend

### Container Registry

Docker images are pushed to GitHub Container Registry (GHCR) using the built-in `GITHUB_TOKEN`. No additional secrets configuration is required. Images will be available at:

- `ghcr.io/hitbhalodiya1/autorec-backend:latest`
- `ghcr.io/hitbhalodiya1/autorec-frontend:latest`

### Node Version

The frontend CI workflow uses Node.js 24 by default. If you need to use Node 20 temporarily, you can set the `ACTIONS_ALLOW_USE_UNSECURE_NODE_VERSION=true` environment variable in the workflow.

## API Documentation

The backend provides interactive API documentation via Swagger UI:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Key Endpoints

#### Datasets
- `POST /api/v1/datasets/upload` - Upload a dataset (CSV/XLSX)
- `GET /api/v1/datasets` - List all datasets
- `GET /api/v1/datasets/{id}` - Get dataset details
- `GET /api/v1/datasets/{id}/preview` - Preview dataset rows
- `POST /api/v1/datasets/{id}/analyze` - Analyze dataset
- `DELETE /api/v1/datasets/{id}` - Delete dataset

#### Experiments
- `POST /api/v1/experiments` - Create experiment
- `GET /api/v1/experiments` - List experiments
- `GET /api/v1/experiments/{id}` - Get experiment details
- `DELETE /api/v1/experiments/{id}` - Delete experiment

#### Preprocessing
- `POST /api/v1/experiments/{id}/preprocessing/recommend` - Get preprocessing recommendations
- `POST /api/v1/experiments/{id}/preprocessing/execute` - Execute preprocessing
- `GET /api/v1/experiments/{id}/preprocessing/status` - Get preprocessing status

#### Training
- `POST /api/v1/experiments/{id}/training/start` - Start training
- `GET /api/v1/experiments/{id}/training/status` - Get training status
- `GET /api/v1/experiments/{id}/training/results` - Get training results

#### Evaluation
- `GET /api/v1/experiments/{id}/evaluation` - Get evaluation summary
- `GET /api/v1/experiments/{id}/evaluation/{model_id}` - Get model evaluation

#### Recommendation
- `GET /api/v1/experiments/{id}/recommendation` - Get model recommendation

#### Explainability
- `POST /api/v1/experiments/{id}/explain/{model_id}` - Get model explanation

## Project Architecture

AutoRec follows a clean architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
│              React + Vite + TailwindCSS                       │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST API
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Layer                                │
│                    FastAPI Routes                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                           │
│              Use Cases & DTOs                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      Domain Layer                             │
│         Entities, Value Objects, Exceptions                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Infrastructure Layer                         │
│    ML Engines, Repositories, Storage, Database                │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

- **Domain Layer**: Pure business logic with no external dependencies
- **Application Layer**: Orchestrates use cases and coordinates domain objects
- **Infrastructure Layer**: Implements interfaces for ML engines, storage, and persistence
- **API Layer**: HTTP endpoints exposing application functionality

## Folder Structure

```
AutoRec/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI routes and middleware
│   │   │   ├── v1/
│   │   │   │   ├── routes/   # API endpoint definitions
│   │   │   │   └── dependencies.py  # Dependency injection
│   │   ├── application/      # Use cases and DTOs
│   │   │   ├── dto/           # Data transfer objects
│   │   │   └── use_cases/    # Business logic orchestration
│   │   ├── core/             # Configuration and constants
│   │   ├── domain/           # Domain entities and value objects
│   │   │   ├── entities/     # Business entities
│   │   │   ├── exceptions/   # Domain exceptions
│   │   │   ├── interfaces/  # Abstract interfaces
│   │   │   └── value_objects/ # Value objects
│   │   └── infrastructure/   # External implementations
│   │       ├── database/     # SQLAlchemy models and repositories
│   │       ├── ml/           # ML engines and registry
│   │       └── storage/      # File storage implementations
│   ├── tests/               # Test suite
│   │   ├── unit/            # Unit tests
│   │   └── integration/     # Integration tests
│   ├── uploads/             # Uploaded datasets
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── pages/           # Page components
│   │   ├── services/        # API clients
│   │   └── stores/          # State management
│   ├── public/              # Static assets
│   ├── Dockerfile
│   └── package.json
├── .github/
│   └── workflows/           # CI/CD configurations
├── docker-compose.yml
└── README.md
```

## Supported Algorithms

### Classification

- **Random Forest**: Gini and Entropy variants
- **Decision Tree**: Gini and Entropy variants
- **Logistic Regression**: With L1/L2 regularization
- **Support Vector Machines**: Linear, RBF, and Polynomial kernels
- **K-Nearest Neighbors**: Configurable neighbors and weights
- **Naive Bayes**: Gaussian variant
- **AdaBoost**: Adaptive boosting
- **Gradient Boosting**: Gradient boosting machines
- **XGBoost**: Extreme gradient boosting
- **Linear Discriminant Analysis**: Dimensionality reduction classifier
- **MLP Classifier**: Multi-layer perceptron neural network

### Regression

- **Linear Regression**: Standard and regularized variants
- **Ridge Regression**: L2 regularization
- **Lasso Regression**: L1 regularization
- **Elastic Net**: Combined L1/L2 regularization
- **Decision Tree Regressor**: Tree-based regression
- **Random Forest Regressor**: Ensemble tree regression
- **Gradient Boosting Regressor**: Boosted tree regression
- **XGBoost Regressor**: Extreme gradient boosting regression
- **AdaBoost Regressor**: Adaptive boosting regression
- **MLP Regressor**: Neural network regression

Each algorithm is configured with multiple parameter variations for comprehensive hyperparameter exploration.

## Screenshots

### Dataset Upload

![Dataset Upload](docs/screenshots/dataset-upload.png)

*Upload CSV or Excel files for analysis and model training*

### Dataset Analysis

![Dataset Analysis](docs/screenshots/dataset-analysis.png)

*Automatic analysis of data quality, feature types, and task detection*

### Experiment Dashboard

![Experiment Dashboard](docs/screenshots/experiment-dashboard.png)

*Manage experiments, track training progress, and view results*

### Model Results

![Model Results](docs/screenshots/model-results.png)

*Compare model performance metrics and rankings*

### Model Explanation

![Model Explanation](docs/screenshots/model-explanation.png)

*Feature importance and SHAP values for model interpretability*

## Future Roadmap

### Phase 1: Enhanced Features
- [ ] Support for time series forecasting
- [ ] Anomaly detection algorithms
- [ ] Clustering and unsupervised learning
- [ ] AutoML hyperparameter optimization
- [ ] Model versioning and A/B testing

### Phase 2: Production Features
- [ ] PostgreSQL and MySQL support
- [ ] Redis caching for performance
- [ ] Celery for async task queues
- [ ] Real-time WebSocket updates
- [ ] Authentication and authorization

### Phase 3: Advanced Capabilities
- [ ] Multi-modal data support (images, text)
- [ ] Deep learning integration (PyTorch, TensorFlow)
- [ ] Distributed training support
- [ ] Model serving and deployment
- [ ] Custom algorithm plugin system

## Contributing

We welcome contributions! Please follow these guidelines:

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- **Backend**: Follow PEP 8, use `ruff` for linting and formatting
- **Frontend**: Follow ESLint rules, use Prettier for formatting
- Write meaningful commit messages
- Add tests for new features
- Update documentation as needed

### Testing

Ensure all tests pass before submitting:

```bash
# Backend
cd backend
pytest tests/ -v --cov=app

# Frontend
cd frontend
npm run test:coverage
```

### Reporting Issues

When reporting bugs, please include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python/Node version)
- Relevant logs or error messages

## License

[Specify your license here]

## Contact

For questions or support, please open an issue on GitHub or contact [your-email@example.com].
