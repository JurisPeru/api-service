import logging
from functools import lru_cache
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    api_key: SecretStr | None = None
    name: str = "claude-3-5-haiku-latest"
    provider: str = "anthropic"


class VectorStoreConfig(BaseModel):
    api_key: SecretStr | None = None
    index_name: str = "02-project-index"
    provider: str = "pinecone"
    rerank_top_n: int = 5


class EmbeddingConfig(BaseModel):
    api_key: SecretStr | None = None
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    size: int = 1536


class LangSmithConfig(BaseModel):
    api_key: SecretStr | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_nested_max_split=1,
    )
    # LLM configs
    llm: LLMConfig = LLMConfig()

    # VectorStore configs
    vector_store: VectorStoreConfig = VectorStoreConfig()

    # Embedding config
    embedding: EmbeddingConfig = EmbeddingConfig()

    # LangSmith config
    langsmith: LangSmithConfig = LangSmithConfig()

    # Logging level
    log_level: str = "INFO"


@lru_cache
def get_settings():
    return Settings()


def setup_logging():
    settings = get_settings()
    level = logging.INFO

    if settings.log_level == "ERROR":
        level = logging.ERROR
    elif settings.log_level == "DEBUG":
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
    )
