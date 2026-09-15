"""Small, dependency-free helpers shared across services."""

import uuid


def generate_document_id() -> str:
    """
    Short, URL-safe, collision-resistant enough for our purposes (not a
    security boundary — just an identifier). 12 hex chars from a UUID4.
    """
    return uuid.uuid4().hex[:12]
