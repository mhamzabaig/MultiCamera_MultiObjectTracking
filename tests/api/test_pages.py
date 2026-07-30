"""Tests for the server-rendered pages.

Deliberately shallow: these routes render a shell and the data arrives client-side,
so what is worth asserting is that every page renders, every asset it references
exists, and the navigation is consistent. Behaviour lives in the API tests, which
exercise the same endpoints the pages call.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from mcmot.ui.pages import PAGES

PATHS = [href for _, href, _ in PAGES]


@pytest.mark.parametrize("path", PATHS)
def test_every_page_renders(client: TestClient, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 200, path
    assert response.headers["content-type"].startswith("text/html")
    assert "<html" in response.text


@pytest.mark.parametrize("path", PATHS)
def test_every_page_links_the_whole_nav(client: TestClient, path: str) -> None:
    """A page that renders but cannot be navigated away from is a dead end."""
    body = client.get(path).text

    for href in PATHS:
        assert f'href="{href}"' in body, f"{path} is missing a link to {href}"


@pytest.mark.parametrize("path", PATHS)
def test_referenced_assets_exist(client: TestClient, path: str) -> None:
    """A 404 on a stylesheet or a script is an unstyled or dead page.

    Worth a test because nothing else would catch a renamed file: the templates
    reference assets by string, so a typo fails silently in the browser.
    """
    body = client.get(path).text
    assets = re.findall(r'(?:href|src)="(/static/[^"]+)"', body)

    assert assets, f"{path} references no static assets at all"
    for asset in assets:
        assert client.get(asset).status_code == 200, f"{path} references missing {asset}"


def test_pages_are_absent_from_the_api_schema(client: TestClient) -> None:
    """The OpenAPI document describes the API; HTML pages are not part of it."""
    paths = client.get("/openapi.json").json()["paths"]

    assert "/registration" not in paths
    assert "/api/persons" in paths


def test_registration_page_warns_about_insecure_origins(client: TestClient) -> None:
    """Browsers block getUserMedia off localhost/HTTPS, so the page must say so.

    Without this the page looks broken rather than blocked, which is the single most
    likely confusion when reaching the app from a phone over plain LAN HTTP.
    """
    body = client.get("/registration").text

    assert "HTTPS" in body
    assert "insecure-warning" in body


def test_tracking_page_points_at_the_stream_endpoint(client: TestClient) -> None:
    body = client.get("/static/js/tracking.js").text

    assert "/stream" in body
