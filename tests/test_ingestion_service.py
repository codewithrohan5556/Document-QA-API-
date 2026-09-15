"""
Tests for GET /health.

Uses httpx.AsyncClient against the app in-process (via ASGITransport) rather
than spinning up a real server — fast, and matches the fact that our routes
are async all the way down.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health_returns_200_and_healthy_status():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
