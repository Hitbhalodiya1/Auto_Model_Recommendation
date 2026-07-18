"""
Main API router — assembles all v1 route modules.
"""

from fastapi import APIRouter

from app.api.v1.routes import datasets, experiments, health, preprocessing, training

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(datasets.router)
api_router.include_router(experiments.router)
api_router.include_router(preprocessing.router)
api_router.include_router(training.router)
