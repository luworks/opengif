"""Explicit configuration for optional database-backed research pipelines."""
import os

def database_dsn() -> str:
    value = os.environ.get("OPENGIF_DATABASE_URL", "").strip()
    if not value:
        raise RuntimeError("Set OPENGIF_DATABASE_URL to your own PostgreSQL database before running a database pipeline")
    return value
