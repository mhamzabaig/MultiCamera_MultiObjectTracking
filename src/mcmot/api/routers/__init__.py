"""HTTP routers.

Phase 9 adds ``persons``, ``cameras``, ``tracking``, ``settings``, and ``logs``
alongside ``system``, each registered in :func:`mcmot.api.routers.all_routers` so
the app factory never needs editing to gain an endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter

from mcmot.api.routers import system

__all__ = ["all_routers"]


def all_routers() -> tuple[APIRouter, ...]:
    """Return every router the application should mount."""
    return (system.router,)
