"""
AutoRec FastAPI application entry point.
Configures middleware, registers routes, and initializes infrastructure on startup.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.middleware.error_handler import (
    domain_exception_handler,
    logging_middleware,
    unhandled_exception_handler,
)
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.domain.exceptions.domain_exceptions import AutoRecError
from app.infrastructure.database.session import create_all_tables, init_database

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Runs startup tasks before yielding and cleanup tasks after.
    """
    settings = get_settings()

    # ── Startup ───────────────────────────────────────────────────────────────
    configure_logging()
    logger.info("autorec_starting", environment=settings.ENVIRONMENT, version=settings.APP_VERSION)

    # Initialize database
    init_database()
    if settings.is_development:
        # Auto-create tables in development (production uses Alembic migrations)
        await create_all_tables()

    # Warm up the model registry
    from app.api.v1.dependencies import get_model_registry
    registry = get_model_registry()
    logger.info("startup_complete", registry_configs=registry.total_configs)

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("autorec_shutting_down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        description="Intelligent Machine Learning Model Recommendation Platform",
        version=settings.APP_VERSION,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        openapi_url="/openapi.json" if settings.is_development else None,
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.middleware("http")(logging_middleware)

    # ── Exception handlers ────────────────────────────────────────────────────
    app.add_exception_handler(AutoRecError, domain_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    return app


app = create_app()
