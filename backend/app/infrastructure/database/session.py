"""
Async SQLAlchemy 2.0 database session management.
Provides engine, session factory, and FastAPI dependency.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    pass


def _build_engine(database_url: str, echo: bool = False) -> AsyncEngine:
    """
    Create the async engine with appropriate settings per database backend.
    SQLite uses StaticPool for single-file access; PostgreSQL uses connection pooling.
    """
    connect_args: dict = {}

    if "sqlite" in database_url:
        # SQLite requires check_same_thread=False for multi-threaded access
        connect_args["check_same_thread"] = False
        return create_async_engine(
            database_url,
            echo=echo,
            connect_args=connect_args,
        )

    settings = get_settings()
    return create_async_engine(
        database_url,
        echo=echo,
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=10,
        pool_pre_ping=True,
    )


# Module-level engine and session factory, initialized at startup
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_database() -> None:
    """
    Initialize the database engine and session factory.
    Called once during application startup.
    """
    global _engine, _session_factory

    settings = get_settings()
    _engine = _build_engine(settings.DATABASE_URL, echo=settings.DATABASE_ECHO)
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )
    logger.info("database_initialized", url=settings.DATABASE_URL.split("///")[0] + "///***")


def get_engine() -> AsyncEngine:
    """Return the initialized engine. Raises if not initialized."""
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the initialized session factory."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session per request.
    Automatically commits on success and rolls back on exception.

    Usage:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_all_tables() -> None:
    """
    Create all tables defined in ORM models.
    Used in development/testing; production uses Alembic migrations.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_tables_created")


async def drop_all_tables() -> None:
    """Drop all tables. Used in testing only."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
