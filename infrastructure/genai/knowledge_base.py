"""Reef knowledge base — ingests documents and builds searchable corpus.

Loads reef science documents (NOAA bulletins, AIMS publications, reef
monitoring guides) into the vector store for RAG retrieval.
"""

from __future__ import annotations

from typing import Any

from infrastructure.genai.embeddings import Embedder, get_embedder, TfidfEmbedder
from infrastructure.genai.vector_store import Document, VectorStore, get_vector_store
from infrastructure.logging import get_logger

logger = get_logger("genai.knowledge_base")

# Sample reef science knowledge — in production, these would be ingested
# from NOAA, AIMS, and Allen Coral Atlas publications.
REEF_KNOWLEDGE_DOCUMENTS = [
    {
        "id": "noaa_dhw_001",
        "content": (
            "Degree Heating Weeks (DHW) is NOAA Coral Reef Watch's primary metric for "
            "cumulative thermal stress on coral reefs. DHW accumulates when the sea surface "
            "temperature (SST) exceeds the Maximum Monthly Mean (MMM) by more than 1°C. "
            "The HotSpot value is defined as SST minus MMM, and only positive HotSpots "
            "above 1°C contribute to DHW. DHW values of 4°C-weeks indicate significant "
            "bleaching is likely, while values above 8°C-weeks indicate widespread bleaching "
            "and significant mortality is expected."
        ),
        "metadata": {"source": "NOAA CRW", "topic": "thermal_stress"},
    },
    {
        "id": "noaa_bleaching_alerts_002",
        "content": (
            "NOAA Coral Reef Watch uses a four-level Bleaching Alert Area system: "
            "No Stress (SST below MMM), Bleaching Watch (SST exceeds MMM but HotSpot < 1°C), "
            "Bleaching Warning (HotSpot >= 1°C but DHW < 4°C-weeks), "
            "Bleaching Alert Level 1 (DHW >= 4°C-weeks, significant bleaching likely), and "
            "Bleaching Alert Level 2 (DHW >= 8°C-weeks, widespread bleaching and mortality expected). "
            "These alerts are based on 5km satellite SST data updated daily."
        ),
        "metadata": {"source": "NOAA CRW", "topic": "bleaching_alerts"},
    },
    {
        "id": "gbr_monitoring_003",
        "content": (
            "The Great Barrier Reef Marine Park Authority and AIMS conduct annual reef health "
            "surveys using the Long-Term Monitoring Program (LTMP). Key indicators include "
            "hard coral cover percentage, soft coral cover, macroalgae cover, and crown-of-thorns "
            "starfish (COTS) density. Hard coral cover above 30% is considered healthy, while "
            "below 10% indicates severe degradation. Recovery from mass bleaching events "
            "typically takes 10-15 years if conditions are favourable."
        ),
        "metadata": {"source": "AIMS", "topic": "reef_monitoring"},
    },
    {
        "id": "bleaching_biology_004",
        "content": (
            "Coral bleaching occurs when symbiotic zooxanthellae (dinoflagellate algae) are "
            "expelled from coral tissues due to stress, primarily thermal stress. The coral-algae "
            "symbiosis is temperature-sensitive: corals live within 1-2°C of their upper thermal "
            "limit. When water temperature exceeds this threshold for sustained periods, the "
            "photosynthetic machinery of zooxanthellae produces reactive oxygen species that "
            "damage coral cells. Bleached coral can recover if stress is removed within weeks, "
            "but prolonged bleaching leads to starvation and mortality."
        ),
        "metadata": {"source": "AIMS", "topic": "bleaching_biology"},
    },
    {
        "id": "water_quality_005",
        "content": (
            "Water quality significantly modulates coral reef resilience. Key parameters include "
            "turbidity (sediment load), nutrient concentrations (nitrogen, phosphorus), and pH. "
            "High turbidity reduces light penetration, limiting photosynthesis by zooxanthellae. "
            "Elevated nutrients promote macroalgae growth that competes with coral. Ocean "
            "acidification (declining pH) reduces coral calcification rates and weakens reef "
            "structures. Dissolved oxygen below 5 mg/L indicates hypoxic conditions that stress "
            "reef organisms. Salinity variations outside 33-36 PSU affect coral osmoregulation."
        ),
        "metadata": {"source": "AIMS", "topic": "water_quality"},
    },
    {
        "id": "reef_resilience_006",
        "content": (
            "Reef resilience refers to the ability of a coral reef ecosystem to resist disturbance "
            "and recover afterwards. Factors promoting resilience include high herbivore fish "
            "populations (which control algae), good water quality, connectivity between reef "
            "patches (for larval recruitment), and genetic diversity in coral populations. "
            "Management strategies to enhance resilience include controlling crown-of-thorns "
            "starfish outbreaks, reducing land-based runoff and nutrient pollution, establishing "
            "marine protected areas, and selective coral breeding programs."
        ),
        "metadata": {"source": "AIMS", "topic": "reef_resilience"},
    },
    {
        "id": "climate_projections_007",
        "content": (
            "Under current emissions trajectories, global sea surface temperatures are projected "
            "to increase by 1.5-3°C by 2100. For tropical coral reefs, this means annual "
            "bleaching events could become the norm by 2040-2050 for most reef regions. "
            "The Great Barrier Reef has experienced mass bleaching in 2016, 2017, 2020, 2022, "
            "and 2024. Each successive event reduces recovery time and cumulative coral cover. "
            "Modelling suggests that limiting warming to 1.5°C would preserve approximately "
            "10-30% of coral reefs, while 2°C warming could reduce this to below 1%."
        ),
        "metadata": {"source": "IPCC/AIMS", "topic": "climate_projections"},
    },
    {
        "id": "digital_twin_concept_008",
        "content": (
            "A coral reef digital twin is a virtual representation that mirrors the state of a "
            "real reef system in near-real-time. Unlike static dashboards, a digital twin "
            "maintains live state variables (temperature, coral cover, stress levels), updates "
            "from observational data streams (sensors, satellites), and supports scenario "
            "simulation ('what if temperature rises 2°C for 3 weeks?'). Key components include "
            "a state engine, data ingestion pipelines, predictive models (bleaching risk, "
            "forecasting), and a serving layer for queries and visualisation."
        ),
        "metadata": {"source": "AIMS/ReefTwin", "topic": "digital_twin"},
    },
    {
        "id": "sst_measurement_009",
        "content": (
            "Sea Surface Temperature (SST) for coral reef monitoring is measured via multiple "
            "sources: satellite radiometers (NOAA AVHRR, MODIS, VIIRS) providing 1-5km "
            "resolution daily data; in-situ temperature loggers deployed on reef structures "
            "recording at 10-minute intervals; and Argo floats measuring subsurface temperature "
            "profiles. NOAA Coral Reef Watch produces daily global 5km SST products by blending "
            "satellite and in-situ data. The Maximum Monthly Mean (MMM) climatology represents "
            "the warmest monthly mean SST expected for a location, and exceedances above "
            "MMM + 1°C trigger bleaching alerts."
        ),
        "metadata": {"source": "NOAA", "topic": "sst_measurement"},
    },
    {
        "id": "reef_restoration_010",
        "content": (
            "Active reef restoration approaches include coral gardening (fragmenting and "
            "replanting coral), assisted gene flow (translocating heat-tolerant genotypes), "
            "larval reseeding (collecting spawn and settling larvae on degraded reefs), and "
            "substrate stabilisation (deploying artificial reef structures). AIMS operates "
            "the National Sea Simulator (SeaSim) facility for selective coral breeding research. "
            "Cost-effectiveness varies: coral gardening costs $10,000-100,000 per hectare, while "
            "natural recovery (if conditions allow) is free but slow (10-15 years)."
        ),
        "metadata": {"source": "AIMS", "topic": "reef_restoration"},
    },
]


class ReefKnowledgeBase:
    """Manages the reef knowledge corpus for RAG retrieval.

    Uses the configured vector store backend (memory, qdrant, pgvector, milvus).
    """

    def __init__(self, embedder: Embedder | None = None, vector_store: VectorStore | None = None) -> None:
        self.embedder = embedder or get_embedder()
        self.vector_store = vector_store or get_vector_store()
        self._is_built = False

    @property
    def is_built(self) -> bool:
        return self._is_built

    def build(self, documents: list[dict[str, Any]] | None = None) -> None:
        """Build the knowledge base from documents."""
        docs = documents or REEF_KNOWLEDGE_DOCUMENTS

        # Fit embedder on corpus (only needed for TF-IDF, no-op for fastembed)
        texts = [d["content"] for d in docs]
        if isinstance(self.embedder, TfidfEmbedder):
            self.embedder.fit(texts)

        # Embed and store
        doc_objects = []
        for d in docs:
            embedding = self.embedder.embed(d["content"])
            doc_objects.append(
                Document(
                    id=d["id"],
                    content=d["content"],
                    metadata=d.get("metadata", {}),
                    embedding=embedding,
                )
            )
        self.vector_store.add_batch(doc_objects)
        self._is_built = True
        logger.info("Knowledge base built with %d documents", len(doc_objects))

    def search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """Search the knowledge base for relevant documents."""
        if not self._is_built:
            raise RuntimeError("Call build() first")

        query_embedding = self.embedder.embed(query)
        results = self.vector_store.search(query_embedding, k=k)

        return [
            {
                "id": r.document.id,
                "content": r.document.content,
                "metadata": r.document.metadata,
                "score": r.score,
            }
            for r in results
        ]


# Module-level singleton
_kb_instance: ReefKnowledgeBase | None = None


def get_knowledge_base() -> ReefKnowledgeBase:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = ReefKnowledgeBase()
        _kb_instance.build()
    return _kb_instance
