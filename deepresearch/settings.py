from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI
    OPENAI_API_KEY: str | None = None
    OpenAI_API_KEY: str | None = None  # compatibility with existing .env

    # Anthropic (Claude)
    ANTHROPIC_API_KEY: str | None = None
    Claude_API_KEY: str | None = None  # compatibility

    # Gemini
    GEMINI_API_KEY: str | None = None
    Gemini_API_KEY: str | None = None  # compatibility

    # PLaMo (OpenAI-compatible)
    PLAMO_BASE_URL: str | None = None
    PLAMO_API_KEY: str | None = None
    PLAMO_MODEL: str | None = None
    PLaMo_API_KEY: str | None = None  # compatibility
    PLaMo_BASE_URL: str | None = None  # compatibility
    PLaMo_MODEL: str | None = None  # compatibility

    # SerpAPI
    SERPAPI_API_KEY: str | None = None

    # Concurrency defaults (balanced)
    LLM_CONCURRENCY_PER_PROVIDER: int = 6
    URL_FETCH_CONCURRENCY: int = 20
    SEARCH_CONCURRENCY: int = 4

    # Storage
    RUNS_DIR: str = "runs"

    def openai_key(self) -> str | None:
        return self.OPENAI_API_KEY or self.OpenAI_API_KEY

    def anthropic_key(self) -> str | None:
        return self.ANTHROPIC_API_KEY or self.Claude_API_KEY

    def gemini_key(self) -> str | None:
        return self.GEMINI_API_KEY or self.Gemini_API_KEY

    def plamo_key(self) -> str | None:
        return self.PLAMO_API_KEY or self.PLaMo_API_KEY

    def plamo_base_url(self) -> str | None:
        return self.PLAMO_BASE_URL or self.PLaMo_BASE_URL

    def plamo_model(self) -> str | None:
        return self.PLAMO_MODEL or self.PLaMo_MODEL

