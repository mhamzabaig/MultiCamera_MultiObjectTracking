"""HTTP routers.

Every router is listed in :func:`all_routers`, so the app factory never needs
editing to gain an endpoint. Order is the order they appear in ``/docs``.
"""

from __future__ import annotations

from fastapi import APIRouter

from mcmot.api.routers import cameras, persons, system, tracking

__all__ = ["all_routers"]


def all_routers() -> tuple[APIRouter, ...]:
    """Return every router the application should mount."""
    return (system.router, persons.router, cameras.router, tracking.router)
