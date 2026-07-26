"""Person image storage.

Images live on disk under ``data/persons/<person_id>/``; the database stores only a
relative path. That keeps the database small enough to copy, makes the images
browsable and backup-friendly, and means the eventual PostgreSQL move does not
have to carry binary columns.

Deliberately byte-oriented rather than array-oriented: registration receives JPEG
bytes from the browser and stores them unchanged, so this module needs no image
library and Phase 2 does not depend on OpenCV.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from mcmot.core.errors import StorageError
from mcmot.core.logging import get_logger

__all__ = ["BlobStore"]

_log = get_logger(__name__)


class BlobStore:
    """Stores and retrieves per-person image files."""

    def __init__(self, root: Path) -> None:
        """Args: root: Directory that holds one subdirectory per person."""
        self._root = root

    @property
    def root(self) -> Path:
        """Root directory of the store."""
        return self._root

    def save(self, person_id: str, filename: str, data: bytes) -> str:
        """Write one image and return its path relative to the store root.

        Args:
            person_id: Owner; becomes the subdirectory name.
            filename: Base name, e.g. ``"01.jpg"``. Any directory component is
                stripped, so a crafted name cannot escape the store.
            data: File contents.

        Returns:
            A relative, forward-slash path suitable for the database, e.g.
            ``"a1b2.../01.jpg"``.

        Raises:
            StorageError: If the name is unusable, the data is empty, or the write
                fails.
        """
        safe_name = Path(filename).name
        if not safe_name or safe_name in {".", ".."}:
            raise StorageError("Invalid image filename.", details={"filename": filename})
        if not data:
            raise StorageError("Refusing to store an empty image.", details={"name": safe_name})

        target = self._root / person_id / safe_name
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        except OSError as exc:
            raise StorageError(
                f"Could not write the image: {target}",
                details={"path": str(target), "reason": str(exc)},
            ) from exc

        return f"{person_id}/{safe_name}"

    def read(self, relative_path: str) -> bytes:
        """Read an image by its stored relative path.

        Raises:
            StorageError: If the file is missing or unreadable.
        """
        target = self.resolve(relative_path)
        try:
            return target.read_bytes()
        except OSError as exc:
            raise StorageError(
                f"Could not read the image: {target}",
                details={"path": relative_path, "reason": str(exc)},
            ) from exc

    def resolve(self, relative_path: str) -> Path:
        """Return the absolute path for a stored relative path.

        Raises:
            StorageError: If the path escapes the store root. A stored value could
                have been edited by hand or, later, arrive from a request, so it is
                checked rather than trusted.
        """
        candidate = (self._root / relative_path).resolve()
        root = self._root.resolve()
        if root != candidate and root not in candidate.parents:
            raise StorageError(
                "Image path escapes the blob directory.",
                details={"path": relative_path},
            )
        return candidate

    def exists(self, relative_path: str) -> bool:
        """Whether the stored path currently resolves to a file."""
        try:
            return self.resolve(relative_path).is_file()
        except StorageError:
            return False

    def delete_person(self, person_id: str) -> int:
        """Remove a person's entire image directory.

        Returns:
            The number of files deleted; ``0`` if the person had no directory.
        """
        directory = self._root / person_id
        if not directory.is_dir():
            return 0
        count = sum(1 for path in directory.rglob("*") if path.is_file())
        try:
            shutil.rmtree(directory)
        except OSError as exc:
            raise StorageError(
                f"Could not delete the image directory: {directory}",
                details={"path": str(directory), "reason": str(exc)},
            ) from exc
        _log.info("blobs.person_deleted", person_id=person_id, files=count)
        return count
