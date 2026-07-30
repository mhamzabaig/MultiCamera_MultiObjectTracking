"""Exception hierarchy for the MC-MOT platform.

Design notes
------------
Every failure the system can produce is a subclass of :class:`MCMOTError` and
carries three things:

``code``
    A stable, machine-readable slug (``camera.disconnected``). The API layer maps
    it straight into an RFC 7807 ``problem+json`` response and the frontend
    switches on it. Because it is declared on the class rather than built from
    the class name, renaming a class never breaks an API contract.
``http_status``
    The transport-level meaning, declared next to the error itself instead of in
    a lookup table inside the API layer that inevitably drifts.
``details``
    Structured context (camera id, path, threshold) for logs. Never free text
    embedded in the message, so it stays queryable in the JSON log stream.

Nothing here imports from elsewhere in the package: this module sits at the
bottom of the dependency graph so any layer may raise from it without cycles.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

__all__ = [
    "CameraAlreadyOpenError",
    "CameraDisconnectedError",
    "CameraError",
    "CameraNotFoundError",
    "CameraTimeoutError",
    "ConfigError",
    "DeviceError",
    "IdentityError",
    "InvalidCameraSourceError",
    "InvalidImageError",
    "MCMOTError",
    "ModelLoadError",
    "PersonNotFoundError",
    "PipelineError",
    "StorageError",
    "VectorIndexError",
]


class MCMOTError(Exception):
    """Base class for every error raised deliberately by this platform.

    Catching :class:`MCMOTError` catches everything the system reports about
    itself, and only that. Anything else escaping a boundary is a genuine bug
    and is logged with a full traceback rather than translated into a tidy
    client response.
    """

    code: str = "internal.error"
    http_status: int = HTTPStatus.INTERNAL_SERVER_ERROR
    #: Human-facing default, used when the raiser supplies no message.
    default_message: str = "An internal error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.details: dict[str, Any] = dict(details or {})
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable payload for API responses and structured logs."""
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "status": int(self.http_status),
        }
        if self.details:
            payload["details"] = self.details
        return payload

    def __repr__(self) -> str:
        """Return a representation carrying the code, for readable test output."""
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class ConfigError(MCMOTError):
    """Configuration is missing, malformed, or internally inconsistent.

    Always raised during startup, never mid-stream: settings are validated once
    and then treated as immutable, so a bad threshold fails the boot instead of
    surfacing as a strange tracking result an hour later.
    """

    code = "config.invalid"
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR
    default_message = "Invalid configuration."


# ---------------------------------------------------------------------------
# Compute device and models
# ---------------------------------------------------------------------------
class DeviceError(MCMOTError):
    """The requested compute device is unavailable and no fallback is permitted.

    Only raised when ``device.allow_cpu_fallback`` is false. With fallback on,
    an unusable GPU is a logged warning and a CPU device, not an error, because
    a degraded system beats a dead one for surveillance workloads.
    """

    code = "device.unavailable"
    http_status = HTTPStatus.SERVICE_UNAVAILABLE
    default_message = "Requested compute device is unavailable."


class ModelLoadError(MCMOTError):
    """Model weights are missing, corrupt, or incompatible with the runtime."""

    code = "model.load_failed"
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR
    default_message = "Failed to load model weights."


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------
class CameraError(MCMOTError):
    """Base class for every camera and stream fault."""

    code = "camera.error"
    http_status = HTTPStatus.BAD_GATEWAY
    default_message = "Camera error."


class CameraNotFoundError(CameraError):
    """No camera is registered under the requested identifier."""

    code = "camera.not_found"
    http_status = HTTPStatus.NOT_FOUND
    default_message = "Camera not found."


class InvalidCameraSourceError(CameraError):
    """The source string is not a recognisable camera.

    A client mistake, not an upstream fault, so it must not inherit
    :class:`CameraError`'s ``502``. Someone typing a phone's stream URL needs to be
    told their URL is malformed, which is a different problem from the camera being
    unreachable — and only the status code carries that distinction to a browser.
    """

    code = "camera.invalid_source"
    http_status = HTTPStatus.UNPROCESSABLE_ENTITY
    default_message = "Unrecognised camera source."


class CameraAlreadyOpenError(CameraError):
    """A camera is already open under this identifier."""

    code = "camera.already_open"
    http_status = HTTPStatus.CONFLICT
    default_message = "A camera is already open under this id."


class CameraDisconnectedError(CameraError):
    """The stream opened successfully but has since dropped.

    Distinct from :class:`CameraTimeoutError` because the recovery policy
    differs: a disconnect triggers the reconnect backoff loop, while a timeout
    may simply mean a slow frame on a congested Wi-Fi link.
    """

    code = "camera.disconnected"
    http_status = HTTPStatus.BAD_GATEWAY
    default_message = "Camera stream disconnected."


class CameraTimeoutError(CameraError):
    """A frame did not arrive within the configured read timeout."""

    code = "camera.timeout"
    http_status = HTTPStatus.GATEWAY_TIMEOUT
    default_message = "Timed out waiting for a camera frame."


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
class StorageError(MCMOTError):
    """Relational storage or the blob store is unavailable or inconsistent."""

    code = "storage.error"
    http_status = HTTPStatus.SERVICE_UNAVAILABLE
    default_message = "Storage backend error."


class PersonNotFoundError(StorageError):
    """No enrolled person exists under the requested identifier."""

    code = "person.not_found"
    http_status = HTTPStatus.NOT_FOUND
    default_message = "Person not found."


class VectorIndexError(MCMOTError):
    """The vector index is unreadable, or has diverged from the database.

    Divergence is recoverable by construction: the index is a derived cache and
    can always be rebuilt from the relational source of truth.
    """

    code = "index.error"
    http_status = HTTPStatus.SERVICE_UNAVAILABLE
    default_message = "Vector index error."


# ---------------------------------------------------------------------------
# Vision pipeline
# ---------------------------------------------------------------------------
class InvalidImageError(MCMOTError):
    """An image or frame could not be decoded, or failed the quality gate."""

    code = "image.invalid"
    http_status = HTTPStatus.UNPROCESSABLE_ENTITY
    default_message = "Image is corrupted or unusable."


class IdentityError(MCMOTError):
    """Enrolment or identity resolution failed."""

    code = "identity.error"
    http_status = HTTPStatus.UNPROCESSABLE_ENTITY
    default_message = "Identity operation failed."


class PipelineError(MCMOTError):
    """A tracking pipeline worker failed to start, or died unexpectedly."""

    code = "pipeline.error"
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR
    default_message = "Tracking pipeline error."
