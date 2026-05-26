"""Hybrid RAG pipeline for reef knowledge retrieval and generation.

Combines dense vector search (TF-IDF/SVD embeddings) with sparse BM25
retrieval, fused via Reciprocal Rank Fusion (RRF). Retrieved context
is passed to Claude for grounded answer generation.

Pattern adapted from llm-twin-enhancements hybrid_search.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from infrastructure.genai.knowledge_base import ReefKnowledgeBase, get_knowledge_base
from infrastructure.genai.llm import generate
from infrastructure.logging import get_logger

logger = get_logger("genai.rag")


@dataclass
class RAGResult:
    answer: str
    sources: list[dict[str, Any]]
    model: str
    input_tokens: int
    output_tokens: int
    retrieval_method: str


class BM25Index:
    """In-memory BM25 sparse retrieval index."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._doc_ids: list[str] = []
        self._doc_contents: list[str] = []

    def build(self, documents: list[dict[str, Any]]) -> None:
        self._doc_ids = [d["id"] for d in documents]
        self._doc_contents = [d["content"] for d in documents]
        tokenized = [doc.lower().split() for doc in self._doc_contents]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built with %d documents", len(documents))

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k (doc_id, score) pairs."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        top_k = min(k, len(self._doc_ids))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_indices]


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists using RRF.

    score(d) = sum(1 / (k + rank_i(d))) for each list i.
    """
    fused_scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    sorted_results = sorted(fused_scores.items(), key=lambda x: -x[1])
    return sorted_results


class HybridRAGPipeline:
    """Hybrid search RAG pipeline: dense + BM25 + RRF + Claude generation."""

    def __init__(self, kb: ReefKnowledgeBase | None = None) -> None:
        self._kb = kb or get_knowledge_base()
        self._bm25 = BM25Index()

        # Build BM25 from knowledge base documents
        docs = self._kb.vector_store.get_all_documents()
        self._bm25.build([
            {"id": d.id, "content": d.content, "metadata": d.metadata}
            for d in docs
        ])
        self._all_docs = {d.id: d for d in docs}

    def retrieve(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """Hybrid retrieval: dense + BM25 + RRF."""
        # Dense search
        dense_results = self._kb.search(query, k=k * 2)
        dense_ranked = [(r["id"], r["score"]) for r in dense_results]

        # Sparse BM25 search
        sparse_ranked = self._bm25.search(query, k=k * 2)

        # Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked])

        # Return top-k with full document content
        results = []
        for doc_id, score in fused[:k]:
            doc = self._all_docs.get(doc_id)
            if doc:
                results.append({
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "rrf_score": score,
                })

        logger.info(
            "Hybrid retrieval for '%s': %d dense + %d sparse → %d fused results",
            query[:50], len(dense_ranked), len(sparse_ranked), len(results),
        )
        return results

    def query(self, question: str, k: int = 3) -> RAGResult:
        """Full RAG pipeline: retrieve context then generate answer."""
        sources = self.retrieve(question, k=k)

        context = "\n\n---\n\n".join(
            f"[Source: {s['metadata'].get('source', 'unknown')} | "
            f"Topic: {s['metadata'].get('topic', 'general')}]\n{s['content']}"
            for s in sources
        )

        system_prompt = (
            "You are a marine science expert specializing in coral reef ecosystems. "
            "Answer questions using the provided context from scientific sources. "
            "Be precise, cite the source when relevant, and note uncertainty when "
            "the context doesn't fully address the question."
        )

        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer based on the context above:"
        )

        response = generate(prompt, system=system_prompt)

        return RAGResult(
            answer=response.content,
            sources=sources,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            retrieval_method="hybrid_rrf",
        )
