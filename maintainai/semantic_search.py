"""Local semantic retrieval with replaceable embedding and index-storage boundaries."""

from dataclasses import dataclass
import hashlib
import json
import sqlite3

import numpy as np

from maintainai.embeddings import EmbeddingError, MiniLMEmbedder, TextEmbedder
from maintainai.service import MaintenanceStore, ValidationError, clean_text, utc_now

TEXT_FIELDS = ("description", "troubleshooting_notes", "root_cause", "corrective_action")


class StaleIndexError(RuntimeError):
    """Search is blocked until the index matches the current ticket snapshot."""


def _signature(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _source_signature(issues: list[dict]) -> str:
    # Include all returned ticket fields and equipment names, not just updated_at.
    return _signature(sorted(issues, key=lambda issue: issue["id"]))


def _parts(issue: dict) -> list[str]:
    return [f"Equipment: {issue['equipment_name']}\n{field.replace('_', ' ').title()}: {issue[field]}"
            for field in TEXT_FIELDS if issue[field].strip()]


def _normalized(values, rows: int, dimension: int) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.shape != (rows, dimension) or not np.isfinite(matrix).all():
            raise ValueError("Invalid shape or non-finite values")
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("Zero-length embedding")
        return (matrix / norms).astype(np.float32)
    except (ValueError, TypeError, OverflowError) as exc:
        raise EmbeddingError("The index contains invalid embeddings. Rebuild the index; check the embedding model if this persists.") from exc


@dataclass(frozen=True)
class IndexStatus:
    current: bool
    total_issues: int
    indexed_issues: int
    built_at: str | None
    reason: str


class SQLiteEmbeddingIndex:
    """Only derived search data is replaced; equipment and tickets are never modified."""

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def read(self) -> tuple[dict | None, list[dict]]:
        metadata = self.connection.execute("SELECT * FROM semantic_index_metadata WHERE id=1").fetchone()
        rows = self.connection.execute("SELECT * FROM issue_embeddings ORDER BY issue_id").fetchall()
        return dict(metadata) if metadata else None, [dict(row) for row in rows]

    def replace(self, rows: list[dict], *, model_key: str, source_signature: str, dimension: int) -> None:
        """Caller owns the transaction, including the source-freshness check."""
        self.connection.execute("DELETE FROM issue_embeddings")
        self.connection.executemany("""INSERT INTO issue_embeddings
            (issue_id, equipment_id, content_hash, dimension, vector)
            VALUES (:issue_id, :equipment_id, :content_hash, :dimension, :vector)""", rows)
        self.connection.execute("""INSERT INTO semantic_index_metadata
            (id, model_key, source_signature, dimension, issue_count, built_at)
            VALUES (1, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET model_key=excluded.model_key,
                source_signature=excluded.source_signature, dimension=excluded.dimension,
                issue_count=excluded.issue_count, built_at=excluded.built_at""",
            (model_key, source_signature, dimension, len(rows), utc_now()))


class SemanticSearch:
    def __init__(self, store: MaintenanceStore, embedder: TextEmbedder | None = None):
        self.store = store
        self.embedder = embedder if embedder is not None else MiniLMEmbedder()
        self.index = SQLiteEmbeddingIndex(store.connection)

    def _assess(self, issues: list[dict], metadata: dict | None, rows: list[dict]) -> IndexStatus:
        reason = "Index is current."
        if metadata is None:
            reason = "Index has not been built."
        elif metadata["model_key"] != self.embedder.model_key or metadata["dimension"] != self.embedder.dimension:
            reason = "The embedding model or index format changed."
        elif metadata["source_signature"] != _source_signature(issues):
            reason = "Tickets or equipment changed since the last rebuild."
        elif metadata["issue_count"] != len(issues) or len(rows) != len(issues):
            reason = "Indexed tickets are missing."
        else:
            by_id = {issue["id"]: issue for issue in issues}
            for row in rows:
                issue = by_id.get(row["issue_id"])
                if (issue is None or row["equipment_id"] != issue["equipment_id"]
                        or row["content_hash"] != _signature(_parts(issue))
                        or row["dimension"] != self.embedder.dimension):
                    reason = "Indexed ticket content does not match the source."
                    break
            if reason == "Index is current." and rows:
                try:
                    self._matrix(rows)
                except (EmbeddingError, ValueError, TypeError):
                    reason = "Stored embeddings are invalid."
        return IndexStatus(reason == "Index is current.", len(issues), len(rows),
                           metadata["built_at"] if metadata else None, reason)

    def _snapshot(self) -> tuple[list[dict], list[dict], IndexStatus]:
        # A read transaction keeps the tickets, vectors and metadata in one snapshot.
        with self.store.connection:
            self.store.connection.execute("BEGIN")
            issues = self.store.list_issues()
            metadata, rows = self.index.read()
            return issues, rows, self._assess(issues, metadata, rows)

    def status(self) -> IndexStatus:
        return self._snapshot()[2]

    def _matrix(self, rows: list[dict]) -> np.ndarray:
        values = [np.frombuffer(row["vector"], dtype="<f4") for row in rows]
        return _normalized(values, len(rows), self.embedder.dimension)

    def rebuild(self) -> int:
        issues = self.store.list_issues()
        source_signature = _source_signature(issues)
        parts = [_parts(issue) for issue in issues]
        texts = [text for group in parts for text in group]
        rows = []
        if texts:
            vectors = _normalized(self.embedder.encode(texts), len(texts), self.embedder.dimension)
            offset = 0
            for issue, group in zip(issues, parts):
                # Encode each populated field separately so a long description cannot
                # crowd troubleshooting notes, root cause or corrective action out.
                pooled = vectors[offset:offset + len(group)].mean(axis=0, keepdims=True)
                vector = _normalized(pooled, 1, self.embedder.dimension)[0]
                rows.append({"issue_id": issue["id"], "equipment_id": issue["equipment_id"],
                             "content_hash": _signature(group), "dimension": self.embedder.dimension,
                             "vector": vector.astype("<f4").tobytes()})
                offset += len(group)
        # Inference happens outside the write transaction. Concurrent ticket edits
        # abort publication of this index and leave the previous index intact.
        with self.store.connection:
            self.store.connection.execute("BEGIN IMMEDIATE")
            if _source_signature(self.store.list_issues()) != source_signature:
                raise StaleIndexError("Tickets changed during indexing. Rebuild again before searching.")
            self.index.replace(rows, model_key=self.embedder.model_key,
                               source_signature=source_signature, dimension=self.embedder.dimension)
        return len(rows)

    def search(self, query: str, *, limit: int = 5) -> list[dict]:
        query = clean_text(query, "Problem description", required=True)
        if type(limit) is not int or not 1 <= limit <= 5:
            raise ValidationError("Search limit must be between 1 and 5.")
        issues, _, status = self._snapshot()
        if not issues:
            return []
        if not status.current:
            raise StaleIndexError(status.reason + " Rebuild the index before searching.")
        query_vector = _normalized(self.embedder.encode([query]), 1, self.embedder.dimension)[0]
        # Check again after model loading/inference; edits during inference cannot
        # produce results labelled current for an earlier ticket snapshot.
        issues, rows, status = self._snapshot()
        if not status.current:
            raise StaleIndexError(status.reason + " Rebuild the index before searching.")
        if not rows:
            return []
        scores = np.clip(self._matrix(rows) @ query_vector, -1.0, 1.0)
        order = np.argsort(-scores, kind="stable")[:limit]
        by_id = {issue["id"]: issue for issue in issues}
        return [{**by_id[rows[index]["issue_id"]], "similarity": float(scores[index])} for index in order]
