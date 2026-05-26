"""Text embedding interface for ReefTwin RAG.

Backends:
    - fastembed: ONNX-based sentence embeddings (default, production quality, no torch)
    - tfidf:     TF-IDF + SVD fallback (no external deps, lightweight)

fastembed uses all-MiniLM-L6-v2 by default (~80MB ONNX model),
providing 384-dimensional embeddings with proper semantic similarity.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import numpy as np

from infrastructure.logging import get_logger

logger = get_logger("genai.embeddings")


class Embedder(ABC):
    """Abstract embedding interface."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return float(np.dot(vec_a, vec_b))


# ---------------------------------------------------------------------------
# FastEmbed Backend (default — production quality)
# ---------------------------------------------------------------------------

class FastEmbedEmbedder(Embedder):
    """ONNX-based sentence embeddings via fastembed.

    Uses all-MiniLM-L6-v2 by default (384 dims, ~80MB).
    No torch required — runs on ONNX Runtime.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name=model_name)
        # Get dim from a test embed
        test = list(self._model.embed(["test"]))[0]
        self._dim = len(test)
        logger.info("FastEmbed loaded: %s (dim=%d)", model_name, self._dim)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        vecs = list(self._model.embed([text]))
        return np.array(vecs[0], dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        vecs = list(self._model.embed(texts))
        return np.array(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# TF-IDF Backend (fallback — no external deps)
# ---------------------------------------------------------------------------

class TfidfEmbedder(Embedder):
    """Lightweight TF-IDF + truncated SVD embedder.

    Fallback when fastembed is not installed. Requires fit() on a corpus.
    """

    def __init__(self, target_dim: int = 128) -> None:
        self._target_dim = target_dim
        self._dim = target_dim
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._svd_components: np.ndarray | None = None
        self._fitted = False

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def fit(self, documents: list[str]) -> None:
        from scipy.sparse.linalg import svds

        n_docs = len(documents)
        if n_docs < 2:
            raise ValueError("Need at least 2 documents to fit")

        doc_freq: dict[str, int] = {}
        tokenized = []
        for doc in documents:
            tokens = self._tokenize(doc)
            tokenized.append(tokens)
            for t in set(tokens):
                doc_freq[t] = doc_freq.get(t, 0) + 1

        filtered = sorted(
            ((t, df) for t, df in doc_freq.items() if df >= 2),
            key=lambda x: -x[1],
        )[:5000]

        self._vocab = {t: i for i, (t, _) in enumerate(filtered)}
        vocab_size = len(self._vocab)

        if vocab_size < self._target_dim:
            self._dim = max(2, vocab_size - 1)

        self._idf = np.zeros(vocab_size)
        for t, i in self._vocab.items():
            self._idf[i] = np.log(n_docs / (1 + doc_freq.get(t, 0)))

        tfidf_matrix = np.zeros((n_docs, vocab_size))
        for doc_idx, tokens in enumerate(tokenized):
            tf: dict[int, float] = {}
            for t in tokens:
                if t in self._vocab:
                    idx = self._vocab[t]
                    tf[idx] = tf.get(idx, 0) + 1
            for idx, count in tf.items():
                tfidf_matrix[doc_idx, idx] = (1 + np.log(count)) * self._idf[idx]

        max_k = min(tfidf_matrix.shape) - 1
        k = min(self._dim, max_k)
        if k < 1:
            self._svd_components = np.eye(vocab_size)
            self._dim = vocab_size
            self._fitted = True
            return

        u, s, vt = svds(np.asarray(tfidf_matrix, dtype=np.float64), k=k)
        self._svd_components = vt
        self._dim = k
        self._fitted = True
        logger.info("TF-IDF embedder fitted: vocab=%d, dim=%d, docs=%d", vocab_size, k, n_docs)

    def embed(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        tokens = self._tokenize(text)
        tfidf = np.zeros(len(self._vocab))
        tf: dict[int, float] = {}
        for t in tokens:
            if t in self._vocab:
                idx = self._vocab[t]
                tf[idx] = tf.get(idx, 0) + 1
        for idx, count in tf.items():
            tfidf[idx] = (1 + np.log(count)) * self._idf[idx]
        vec = tfidf @ self._svd_components.T
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedder(backend: str = "fastembed") -> Embedder:
    """Create an embedder. Falls back to TF-IDF if fastembed unavailable."""
    if backend == "fastembed":
        try:
            return FastEmbedEmbedder()
        except ImportError:
            logger.warning("fastembed not installed, falling back to TF-IDF")
            return TfidfEmbedder()
    elif backend == "tfidf":
        return TfidfEmbedder()
    else:
        raise ValueError(f"Unknown embedding backend: {backend!r}. Options: fastembed, tfidf")
