"""Tests for GenAI components (work without ANTHROPIC_API_KEY)."""

import numpy as np
import pytest

from infrastructure.genai.embeddings import TfidfEmbedder
from infrastructure.genai.vector_store import Document, InMemoryVectorStore, VectorStore, get_vector_store
from infrastructure.genai.knowledge_base import ReefKnowledgeBase, REEF_KNOWLEDGE_DOCUMENTS
from infrastructure.genai.rag import BM25Index, reciprocal_rank_fusion, HybridRAGPipeline
from infrastructure.genai.router import QueryRouter, QueryComplexity
from infrastructure.genai.agent import build_reef_tools
from infrastructure.genai.llm import generate, LLMResponse, get_provider, MockProvider, LLMProvider


# --- Embeddings ---

def test_tfidf_embedder_fit_and_embed():
    docs = [
        "coral bleaching is caused by thermal stress",
        "reef monitoring uses satellite temperature data",
        "water quality affects coral health and resilience",
        "degree heating weeks measure cumulative thermal exposure",
    ]
    embedder = TfidfEmbedder(target_dim=8)
    embedder.fit(docs)

    vec = embedder.embed("thermal stress on coral")
    assert vec.shape[0] > 0
    # Should be normalized
    assert abs(np.linalg.norm(vec) - 1.0) < 0.01


def test_tfidf_embedder_similarity():
    # Need enough docs and vocabulary for SVD to produce meaningful results
    docs = REEF_KNOWLEDGE_DOCUMENTS[:5]
    texts = [d["content"] for d in docs]
    embedder = TfidfEmbedder(target_dim=16)
    embedder.fit(texts)

    vec_thermal = embedder.embed("coral bleaching thermal stress degree heating weeks")
    vec_water = embedder.embed("water quality turbidity sediment nutrients")

    # Both should produce valid normalized vectors
    assert abs(np.linalg.norm(vec_thermal) - 1.0) < 0.1
    assert abs(np.linalg.norm(vec_water) - 1.0) < 0.1


def test_tfidf_batch_embed():
    docs = REEF_KNOWLEDGE_DOCUMENTS  # use full corpus for stable dim
    texts = [d["content"] for d in docs]
    embedder = TfidfEmbedder(target_dim=16)
    embedder.fit(texts)
    batch = embedder.embed_batch(["coral reef bleaching", "ocean temperature monitoring"])
    assert batch.shape[0] == 2
    assert batch.shape[1] == embedder.dim


# --- Vector Store ---

def test_vector_store_add_and_search():
    store = InMemoryVectorStore()
    docs = [
        Document(id="a", content="coral bleaching", embedding=np.array([1.0, 0.0, 0.0])),
        Document(id="b", content="reef monitoring", embedding=np.array([0.0, 1.0, 0.0])),
        Document(id="c", content="water quality", embedding=np.array([0.0, 0.0, 1.0])),
    ]
    store.add_batch(docs)
    assert store.count == 3

    results = store.search(np.array([1.0, 0.0, 0.0]), k=2)
    assert len(results) == 2
    assert results[0].document.id == "a"
    assert results[0].score > results[1].score


def test_vector_store_empty_search():
    store = InMemoryVectorStore()
    results = store.search(np.array([1.0, 0.0]), k=3)
    assert results == []


def test_vector_store_clear():
    store = InMemoryVectorStore()
    store.add(Document(id="x", content="test", embedding=np.array([1.0])))
    assert store.count == 1
    store.clear()
    assert store.count == 0


# --- Knowledge Base ---

def test_knowledge_base_build_and_search():
    kb = ReefKnowledgeBase(embedder=TfidfEmbedder(target_dim=16))
    kb.build()
    assert kb.is_built

    results = kb.search("degree heating weeks thermal stress", k=3)
    assert len(results) > 0
    assert "content" in results[0]
    assert "score" in results[0]


def test_knowledge_base_documents_count():
    kb = ReefKnowledgeBase(embedder=TfidfEmbedder(target_dim=16))
    kb.build()
    assert kb.vector_store.count == len(REEF_KNOWLEDGE_DOCUMENTS)


# --- BM25 ---

def test_bm25_index_build_and_search():
    bm25 = BM25Index()
    docs = [
        {"id": "d1", "content": "coral bleaching thermal stress degree heating weeks"},
        {"id": "d2", "content": "reef fish population biodiversity marine reserve"},
        {"id": "d3", "content": "sea surface temperature satellite monitoring"},
    ]
    bm25.build(docs)
    results = bm25.search("thermal stress bleaching", k=2)
    assert len(results) == 2
    assert results[0][0] == "d1"  # Should rank the thermal doc first


# --- Reciprocal Rank Fusion ---

def test_rrf_fusion():
    list_a = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
    list_b = [("doc2", 0.8), ("doc1", 0.6), ("doc4", 0.4)]

    fused = reciprocal_rank_fusion([list_a, list_b])
    doc_ids = [d[0] for d in fused]
    # doc1 and doc2 appear in both lists, should be ranked high
    assert "doc1" in doc_ids[:3]
    assert "doc2" in doc_ids[:3]


# --- Hybrid RAG ---

def test_hybrid_rag_retrieve():
    kb = ReefKnowledgeBase(embedder=TfidfEmbedder(target_dim=16))
    kb.build()
    pipeline = HybridRAGPipeline(kb=kb)

    results = pipeline.retrieve("what causes coral bleaching", k=3)
    assert len(results) > 0
    assert "rrf_score" in results[0]


def test_hybrid_rag_query_mock():
    """RAG query returns a mock response without API key."""
    kb = ReefKnowledgeBase(embedder=TfidfEmbedder(target_dim=16))
    kb.build()
    pipeline = HybridRAGPipeline(kb=kb)

    result = pipeline.query("what is degree heating weeks")
    assert result.answer  # Should get mock response
    assert len(result.sources) > 0
    assert result.retrieval_method == "hybrid_rrf"


# --- Router ---

def test_router_simple_query():
    router = QueryRouter()
    decision = router.route("What is the current state of gbr_heron_reef?")
    assert decision.complexity == QueryComplexity.SIMPLE
    assert decision.handler == "api_lookup"


def test_router_moderate_query():
    router = QueryRouter()
    decision = router.route("How does ocean acidification affect coral reef resilience?")
    assert decision.complexity == QueryComplexity.MODERATE
    assert decision.handler == "rag"


def test_router_complex_query():
    router = QueryRouter()
    decision = router.route(
        "Compare the bleaching risk across all reefs and recommend intervention "
        "strategies based on the current conditions and projected scenarios"
    )
    assert decision.complexity == QueryComplexity.COMPLEX
    assert decision.handler == "agent"


# --- Agent Tools ---

def test_agent_tools_defined():
    tools = build_reef_tools()
    assert len(tools) == 4
    names = {t.name for t in tools}
    assert "query_reef_state" in names
    assert "run_simulation" in names
    assert "search_knowledge_base" in names
    assert "get_stress_breakdown" in names


def test_tool_claude_format():
    tools = build_reef_tools()
    for tool in tools:
        schema = tool.to_claude_tool()
        assert "name" in schema
        assert "description" in schema
        assert "input_schema" in schema


# --- LLM mock mode ---

def test_llm_mock_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Reset the cached client
    import infrastructure.genai.llm as llm_mod
    llm_mod._client = None

    response = generate("test prompt")
    assert isinstance(response, LLMResponse)
    assert response.model == "mock"
    assert "Mock" in response.content


# --- API Endpoints ---

# --- Pluggable LLM Providers ---

def test_mock_provider():
    provider = MockProvider()
    assert provider.provider_name == "mock"
    response = provider.generate("test")
    assert isinstance(response, LLMResponse)
    assert response.provider == "mock"


def test_get_provider_fallback_to_mock(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import infrastructure.genai.llm as llm_mod
    llm_mod._provider = None
    provider = get_provider("claude")  # Should fall back to mock
    assert isinstance(provider, MockProvider)
    llm_mod._provider = None  # Reset


def test_llm_provider_interface():
    """All providers must implement the LLMProvider interface."""
    assert issubclass(MockProvider, LLMProvider)


# --- Pluggable Vector Store ---

def test_vector_store_factory_memory():
    store = get_vector_store("memory")
    assert store.backend_name == "memory"


def test_vector_store_abstract_interface():
    """InMemoryVectorStore must implement VectorStore."""
    assert issubclass(InMemoryVectorStore, VectorStore)
    store = InMemoryVectorStore()
    assert hasattr(store, "backend_name")
    assert hasattr(store, "count")
    assert hasattr(store, "add")
    assert hasattr(store, "add_batch")
    assert hasattr(store, "search")
    assert hasattr(store, "get_all_documents")
    assert hasattr(store, "clear")


def test_vector_store_factory_unknown():
    with pytest.raises(ValueError, match="Unknown vector store"):
        get_vector_store("nonexistent")


# --- API Endpoints ---

def test_rag_endpoint():
    from fastapi.testclient import TestClient
    from services.twin_api.main import app
    client = TestClient(app)
    response = client.post("/rag", json={"question": "what is DHW"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data


def test_query_endpoint_simple():
    from fastapi.testclient import TestClient
    from services.twin_api.main import app
    client = TestClient(app)
    response = client.post("/query", json={"query": "list reefs"})
    assert response.status_code == 200
    data = response.json()
    assert "routing" in data
