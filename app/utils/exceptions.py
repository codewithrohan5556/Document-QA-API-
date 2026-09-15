"""
Application-specific exceptions.

Each carries an HTTP status code. main.py registers a single exception
handler for `DocumentQAError` that converts any of these into a clean JSON
body — `{"error": "..."}`` — instead of a raw traceback. Routers/services
just `raise` the appropriate one and don't touch HTTP concerns directly.
"""


class DocumentQAError(Exception):
    """Base class for all application errors that should become clean JSON responses."""

    status_code = 400

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class UnsupportedFileTypeError(DocumentQAError):
    status_code = 415


class EmptyDocumentError(DocumentQAError):
    status_code = 400


class DocumentNotFoundError(DocumentQAError):
    status_code = 404


class VectorStoreError(DocumentQAError):
    status_code = 502


class LLMError(DocumentQAError):
    status_code = 502


class ServiceUnavailableError(DocumentQAError):
    status_code = 503
