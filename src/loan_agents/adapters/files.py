"""Uploaded documents and contract drafts on disk: <root>/<case_id>/<doc_id>.<ext>."""

from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path

from loan_agents.contracts import CASE_ID_PATTERN

_EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "text/markdown": "md",
}
_MIME_BY_EXTENSION = {extension: mime for mime, extension in _EXTENSIONS.items()}
_DOC_ID = re.compile(r"^[0-9a-f]{32}$")
_TEXT_NAME = re.compile(r"^[a-z0-9-]{1,40}\.md$")


class FileDocumentStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def _case_dir(self, case_id: str) -> Path:
        # Case ids end up in paths, so they are checked here as well as in the service.
        if not CASE_ID_PATTERN.match(case_id):
            raise ValueError(f"invalid case id {case_id!r}")
        return self._root / case_id

    async def save(self, case_id: str, data: bytes, mime_type: str) -> str:
        extension = _EXTENSIONS.get(mime_type)
        if extension is None:
            raise ValueError(f"unsupported mime type {mime_type!r}")
        doc_id = uuid.uuid4().hex
        path = self._case_dir(case_id) / f"{doc_id}.{extension}"
        await asyncio.to_thread(_write_bytes, path, data)
        return doc_id

    async def load(self, case_id: str, doc_id: str) -> tuple[bytes, str]:
        if not _DOC_ID.match(doc_id):
            raise ValueError(f"invalid document id {doc_id!r}")
        matches = sorted(self._case_dir(case_id).glob(f"{doc_id}.*"))
        if not matches:
            raise FileNotFoundError(f"case {case_id}: no document {doc_id}")
        path = matches[0]
        mime_type = _MIME_BY_EXTENSION[path.suffix.removeprefix(".")]
        return await asyncio.to_thread(path.read_bytes), mime_type

    async def save_text(self, case_id: str, name: str, text: str) -> str:
        if not _TEXT_NAME.match(name):
            raise ValueError(f"invalid file name {name!r}")
        doc_id = uuid.uuid4().hex
        path = self._case_dir(case_id) / f"{doc_id}.md"
        await asyncio.to_thread(_write_bytes, path, text.encode("utf-8"))
        return doc_id


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if path.stat().st_size != len(data):
        raise OSError(f"{path}: wrote {path.stat().st_size} of {len(data)} bytes")
