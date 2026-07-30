"""HTML page routes.

Every handler does the same thing: render a shell that names the active nav item.
The data arrives client-side from the REST API, so these routes cannot drift from
what the API reports.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

__all__ = ["STATIC_DIR", "router"]

_UI_DIR = Path(__file__).resolve().parent
STATIC_DIR = _UI_DIR / "static"
_templates = Jinja2Templates(directory=str(_UI_DIR / "templates"))

#: Nav order, and the single source of truth for page titles.
PAGES: tuple[tuple[str, str, str], ...] = (
    ("dashboard", "/", "Dashboard"),
    ("registration", "/registration", "Registration"),
    ("tracking", "/tracking", "Tracking"),
    ("database", "/database", "Database"),
    ("cameras", "/cameras", "Cameras"),
    ("settings", "/settings", "Settings"),
    ("logs", "/logs", "Logs"),
)

router = APIRouter(include_in_schema=False)


def _render(request: Request, page: str) -> HTMLResponse:
    """Render one page's template with the shared navigation context."""
    title = next(label for slug, _, label in PAGES if slug == page)
    return _templates.TemplateResponse(
        request=request,
        name=f"{page}.html",
        context={"page": page, "title": title, "pages": PAGES},
    )


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """System overview: health, gallery size, cameras, running pipelines."""
    return _render(request, "dashboard")


@router.get("/registration", response_class=HTMLResponse)
async def registration(request: Request) -> HTMLResponse:
    """Enrol a person using this device's camera."""
    return _render(request, "registration")


@router.get("/tracking", response_class=HTMLResponse)
async def tracking(request: Request) -> HTMLResponse:
    """Live annotated video with per-camera metrics."""
    return _render(request, "tracking")


@router.get("/database", response_class=HTMLResponse)
async def database(request: Request) -> HTMLResponse:
    """The enrolled gallery: browse, rename, delete, rebuild the index."""
    return _render(request, "database")


@router.get("/cameras", response_class=HTMLResponse)
async def cameras(request: Request) -> HTMLResponse:
    """Add and remove cameras, and watch their connection state."""
    return _render(request, "cameras")


@router.get("/settings", response_class=HTMLResponse)
async def settings(request: Request) -> HTMLResponse:
    """The active configuration, read-only."""
    return _render(request, "settings")


@router.get("/logs", response_class=HTMLResponse)
async def logs(request: Request) -> HTMLResponse:
    """Tail of the structured log."""
    return _render(request, "logs")
