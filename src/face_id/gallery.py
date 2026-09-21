"""On-disk enrolled identity store (JSON metadata + NPZ embeddings)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from face_id.config import EMBEDDING_DIM, GALLERY_DIR


class Gallery:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or GALLERY_DIR)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.json"
        self._embeddings_path = self.root / "embeddings.npz"
        self._index = self._load_index()
        self._embeddings = self._load_embeddings()

    def _load_index(self) -> dict:
        if not self._index_path.exists():
            return {"identities": {}}
        return json.loads(self._index_path.read_text(encoding="utf-8"))

    def _load_embeddings(self) -> dict[str, np.ndarray]:
        if not self._embeddings_path.exists():
            return {}
        data = np.load(self._embeddings_path, allow_pickle=False)
        return {key: data[key] for key in data.files}

    def reload(self) -> None:
        self._index = self._load_index()
        self._embeddings = self._load_embeddings()

    def _save(self) -> None:
        self._index_path.write_text(
            json.dumps(self._index, indent=2), encoding="utf-8"
        )
        if self._embeddings:
            np.savez_compressed(self._embeddings_path, **self._embeddings)
        elif self._embeddings_path.exists():
            self._embeddings_path.unlink()

    def identities(self) -> list[str]:
        return sorted(self._index["identities"].keys())

    def as_dict(self) -> dict[str, list[np.ndarray]]:
        out: dict[str, list[np.ndarray]] = {}
        for name, matrix in self._embeddings.items():
            out[name] = [matrix[i] for i in range(matrix.shape[0])]
        return out

    def enroll(
        self,
        identity: str,
        embedding: np.ndarray,
        source: str | None = None,
    ) -> int:
        identity = identity.strip()
        if not identity:
            raise ValueError("Please type a name.")
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.shape[0] != EMBEDDING_DIM:
            raise ValueError(
                f"expected {EMBEDDING_DIM}-d embedding, got {vector.shape[0]}"
            )
        existing = self._embeddings.get(identity)
        if existing is None:
            stacked = vector.reshape(1, -1)
        else:
            stacked = np.vstack([existing, vector])
        self._embeddings[identity] = stacked
        record = self._index["identities"].setdefault(
            identity, {"templates": 0, "sources": []}
        )
        record["templates"] = int(stacked.shape[0])
        record["sources"].append(
            {
                "path": source,
                "enrolled_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._save()
        return int(stacked.shape[0])

    def remove(self, identity: str) -> None:
        self._index["identities"].pop(identity, None)
        self._embeddings.pop(identity, None)
        self._save()

    def clear(self) -> None:
        self._index = {"identities": {}}
        self._embeddings = {}
        self._save()

    def template_count(self, identity: str) -> int:
        matrix = self._embeddings.get(identity)
        return 0 if matrix is None else int(matrix.shape[0])
