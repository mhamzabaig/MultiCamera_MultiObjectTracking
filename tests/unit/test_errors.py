"""Exception hierarchy tests.

Error codes are an API contract: the frontend switches on them and operators grep
for them. Pinning them in a test means a future rename of a class cannot silently
change the wire format.
"""

from __future__ import annotations

from http import HTTPStatus

import pytest

from mcmot.core.errors import (
    CameraDisconnectedError,
    CameraError,
    CameraNotFoundError,
    CameraTimeoutError,
    ConfigError,
    DeviceError,
    IdentityError,
    InvalidImageError,
    MCMOTError,
    ModelLoadError,
    PersonNotFoundError,
    PipelineError,
    StorageError,
    VectorIndexError,
)

ALL_ERRORS = [
    ConfigError,
    DeviceError,
    ModelLoadError,
    CameraError,
    CameraNotFoundError,
    CameraDisconnectedError,
    CameraTimeoutError,
    StorageError,
    PersonNotFoundError,
    VectorIndexError,
    InvalidImageError,
    IdentityError,
    PipelineError,
]


class TestHierarchy:
    @pytest.mark.parametrize("error_cls", ALL_ERRORS)
    def test_every_error_derives_from_the_base(self, error_cls: type[MCMOTError]) -> None:
        assert issubclass(error_cls, MCMOTError)
        assert issubclass(error_cls, Exception)

    @pytest.mark.parametrize(
        "error_cls", [CameraNotFoundError, CameraDisconnectedError, CameraTimeoutError]
    )
    def test_camera_faults_share_a_catchable_base(self, error_cls: type[CameraError]) -> None:
        """One `except CameraError` must cover every stream fault."""
        with pytest.raises(CameraError):
            raise error_cls

    def test_person_not_found_is_a_storage_error(self) -> None:
        with pytest.raises(StorageError):
            raise PersonNotFoundError


class TestContract:
    @pytest.mark.parametrize(
        ("error_cls", "code", "status"),
        [
            (ConfigError, "config.invalid", HTTPStatus.INTERNAL_SERVER_ERROR),
            (DeviceError, "device.unavailable", HTTPStatus.SERVICE_UNAVAILABLE),
            (ModelLoadError, "model.load_failed", HTTPStatus.INTERNAL_SERVER_ERROR),
            (CameraNotFoundError, "camera.not_found", HTTPStatus.NOT_FOUND),
            (CameraDisconnectedError, "camera.disconnected", HTTPStatus.BAD_GATEWAY),
            (CameraTimeoutError, "camera.timeout", HTTPStatus.GATEWAY_TIMEOUT),
            (PersonNotFoundError, "person.not_found", HTTPStatus.NOT_FOUND),
            (VectorIndexError, "index.error", HTTPStatus.SERVICE_UNAVAILABLE),
            (InvalidImageError, "image.invalid", HTTPStatus.UNPROCESSABLE_ENTITY),
        ],
    )
    def test_codes_and_statuses_are_pinned(
        self, error_cls: type[MCMOTError], code: str, status: HTTPStatus
    ) -> None:
        assert error_cls.code == code
        assert error_cls.http_status == status

    def test_codes_are_unique(self) -> None:
        codes = [cls.code for cls in ALL_ERRORS]
        assert len(codes) == len(set(codes))

    def test_default_message_is_used_when_none_supplied(self) -> None:
        assert CameraTimeoutError().message == "Timed out waiting for a camera frame."

    def test_explicit_message_wins(self) -> None:
        assert CameraTimeoutError("cam-3 stalled").message == "cam-3 stalled"


class TestSerialisation:
    def test_to_dict_contains_the_wire_contract(self) -> None:
        error = CameraDisconnectedError(
            "Stream dropped", details={"camera_id": "cam-1", "attempts": 3}
        )

        payload = error.to_dict()

        assert payload == {
            "code": "camera.disconnected",
            "message": "Stream dropped",
            "status": int(HTTPStatus.BAD_GATEWAY),
            "details": {"camera_id": "cam-1", "attempts": 3},
        }

    def test_details_are_omitted_when_empty(self) -> None:
        assert "details" not in StorageError().to_dict()

    def test_details_are_copied_not_aliased(self) -> None:
        """A caller mutating its dict afterwards must not alter the raised error."""
        source = {"camera_id": "cam-1"}
        error = CameraError(details=source)

        source["camera_id"] = "cam-2"

        assert error.details["camera_id"] == "cam-1"

    def test_repr_is_diagnostic(self) -> None:
        assert "camera.timeout" in repr(CameraTimeoutError("slow link"))
