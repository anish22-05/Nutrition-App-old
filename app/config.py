from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration lives here. Values come from env vars / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # secrets (required)
    openai_api_key: str
    supabase_url: str
    supabase_service_role_key: str

    # identity. Set AUTH_REQUIRED=true for anything public (see app/auth.py).
    auth_required: bool = False
    supabase_anon_key: str = ""  # used by the Streamlit login form, never by the API

    # models
    embed_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"

    # uploads. Core documents are added by the admin CLI (generous limits); user uploads are
    # capped because every upload costs real embedding money and anyone can hit the endpoint.
    max_upload_mb: int = 50           # core / admin
    admin_max_pages: int = 3000
    user_max_upload_mb: int = 25
    user_max_pages: int = 400
    user_max_chunks: int = 1500
    user_max_docs: int = 5
    min_text_page_ratio: float = 0.3  # share of pages that must yield text, else it is a scan
    max_garble_ratio: float = 0.05    # share of junk characters tolerated before we call it garbled

    # abuse protection (requests per minute, per client IP and per owner id)
    rate_limit_chat_per_min: int = 20
    rate_limit_upload_per_min: int = 5

    # ingestion / chunking
    sents_per_chunk: int = 10
    sent_overlap: int = 2
    max_chunk_tokens: int = 1300
    min_chunk_tokens: int = 30
    embed_batch_size: int = 100
    insert_batch_size: int = 50

    # retrieval
    default_top_k: int = 5
    max_top_k: int = 20
    min_similarity: float = 0.35
    retrieval_mode: str = "vector"      # vector | hybrid (see eval results for why)
    rerank_enabled: bool = False
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 30

    # Section-aware chunks: embed "Chapter > Section" in front of each chunk (from the PDF outline).
    # Off by default: no measurable retrieval gain on the textbook (README, Phase 6). Section titles
    # are recorded in chunk metadata and shown with sources either way.
    contextual_embeddings: bool = False
    # Split multi-part questions ("compare X and Y") into sub-queries, retrieve each, merge.
    # Off by default: it *lowered* recall on the golden set (README, Phase 6). Opt-in.
    query_decomposition: bool = False
    max_subqueries: int = 3

    # storage: "supabase" (production) or "local" (a file on disk; no database needed - dev/demo)
    vector_store: str = "supabase"
    local_store_path: str = "data/local_store.pkl"

    # tools the answering model may call (app/tools.py)
    tools_enabled: bool = True
    usda_api_key: str = "DEMO_KEY"  # FoodData Central; DEMO_KEY is rate limited (get a free key at api.data.gov)
    max_tool_rounds: int = 3

    # answer cache: reuse an answer when a new question means the same thing (cosine >= threshold)
    cache_enabled: bool = True
    cache_similarity: float = 0.97
    cache_ttl_s: int = 3600
    cache_max_entries: int = 500

    # store each answered question (+ thumbs up/down) for review and for growing the eval set
    log_interactions: bool = True

    # evaluation (see eval/)
    judge_model: str = "gpt-4o-mini"


@lru_cache
def get_settings() -> Settings:
    return Settings()
