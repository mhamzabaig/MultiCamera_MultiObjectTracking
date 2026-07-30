"""Server-rendered pages and their static assets.

Thin by design. Each page is a Jinja shell that fetches its data from the same
REST API any other client would use, so there is exactly one path from the
services to a screen. A page that rendered data server-side would be a second,
silently divergent view of the same state.

**No CDN, no framework, no build step.** Bootstrap, HTMX, and Alpine were the
original plan; the pages ended up needing camera capture, an ``<img>`` tag, and
``fetch`` — none of which a framework helps with. Loading three libraries from a
CDN would have meant that a phone on a flaky connection, or a laptop with no
internet at all, gets an unstyled page for a system whose whole point is running
on your own LAN. The stylesheet and scripts here are served from the same origin
as everything else and work offline.
"""

from __future__ import annotations

from mcmot.ui.pages import STATIC_DIR, router

__all__ = ["STATIC_DIR", "router"]
