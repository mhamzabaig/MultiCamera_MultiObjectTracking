"""FastAPI dependency providers.

Handlers receive their collaborators as parameters rather than reaching for a
module-level global. The container is stored on ``app.state`` at startup, so a
test can build an app around a container wired with fakes and every handler
follows automatically.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from mcmot.core.container import Container
from mcmot.core.settings import Settings

__all__ = ["ContainerDep", "SettingsDep", "get_container", "get_settings"]


def get_container(request: Request) -> Container:
    """Return the application container attached to the running app.

    Raises:
        RuntimeError: If the app was constructed without a container. That is a
            wiring bug, not a client error, so it is not an ``MCMOTError``.
    """
    container: Container | None = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - guards against a wiring mistake
        raise RuntimeError(
            "Application container is not initialised; build the app with create_app()."
        )
    return container


def get_settings(container: Annotated[Container, Depends(get_container)]) -> Settings:
    """Return the active configuration."""
    return container.settings


ContainerDep = Annotated[Container, Depends(get_container)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
