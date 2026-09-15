"""
Tests for cross-cutting error handling and general API robustness (Step 16
coverage pass): the generic exception handler, standard HTTP behavior for
unknown routes/methods, and that the OpenAPI schema itself is valid (a
broken Pydantic response model would otherwise only surface when someone
happens to load /docs).
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_unhandled_exception_returns_clean_500_json(monkeypatch):
    """
    Proves the catch-all handler added in Step 16: a bug that raises a
    plain Python exception (not one of our DocumentQAError subclasses)
    still becomes a clean JSON error, never a raw traceback.

    raise_app_exceptions=False is required here: httpx's ASGITransport
    re-raises app-level exceptions into the test by default, as a debugging
    convenience — even when the app's own exception handler already
    produced a valid response. A real ASGI server (uvicorn) does NOT do
    this; the client only ever sees the JSON response. Disabling it here
    makes the test observe what a real client would actually receive.
    """
    def _boom():
        raise KeyError("something unexpected broke")

    monkeypatch.setattr("app.routers.documents.get_vector_store", _boom)

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/stats")

    assert response.status_code == 500
    body = response.json()
    assert "error" in body
    # No raw traceback text should ever reach the client.
    assert "Traceback" not in response.text
    assert "KeyError" not in response.text


@pytest.mark.asyncio
async def test_unknown_route_returns_404():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/this-route-does-not-exist")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_wrong_http_method_returns_405():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/upload")  # /upload is POST-only

    assert response.status_code == 405


@pytest.mark.asyncio
async def test_openapi_schema_is_valid_and_lists_every_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/upload" in paths
    assert "/query" in paths
    assert "/stats" in paths
    assert "/health" in paths
    assert "/document/{document_id}" in paths
    assert "/conversations/{session_id}" in paths
