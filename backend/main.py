"""
main.py
FastAPI application entry point.
Phase 1 — health endpoint only.
Endpoints added per phase as agents are built.
"""

from fastapi import FastAPI
from backend.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AML Investigation Agent",
    description="Autonomous AML investigation system with multi-agent validation", version="0.1.0")


@app.on_event("startup")
async def startup_event():
    logger.info("AML Investigation Agent starting up...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Confidence threshold: {settings.confidence_threshold}")
    logger.info(f"Qdrant host: {settings.qdrant_host}:{settings.qdrant_port}")


@app.get("/health")
async def health_check():
    """Health check endpoint — confirms service is running."""
    return {
        "status": "ok",
        "service": "aml-investigation-agent",
        "version": "0.1.0",
        "environment": settings.environment
    }


@app.get("/")
async def root():
    """Root endpoint — redirects to docs."""
    return {
        "message": "AML Investigation Agent API",
        "docs": "/docs",
        "health": "/health"
    }