from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache


# app settings class
class Settings(BaseSettings):
    # general api config
    PROJECT_NAME: str = "Vietnamese RAG System"
    API_V1_STR: str = "/api/v1"

    # qdrant vector db
    QDRANT_HOST: str = Field(default="localhost")
    QDRANT_PORT: int = Field(default=6333)

    # redis cache status
    REDIS_URL: str = Field(default="redis://localhost:6379")

    # --- Generator config ---
    # Use the same Gemma base model as the external-RAG candidate in
    # trustmargin-experiments.  The token is needed only when the model is
    # gated or has not already been cached locally by Hugging Face.
    HF_MODEL_NAME: str = Field(default="google/gemma-2-2b-it")
    HF_TOKEN: str | None = Field(default=None)
    GEMMA_DEVICE: str = Field(default="cuda")
    GEMMA_MAX_INPUT_TOKENS: int = Field(default=4096, ge=128)
    LLM_MAX_NEW_TOKENS: int = Field(default=256, ge=1, le=2048)
    RAG_MAX_CONTEXT_CHARS: int = Field(default=12000, ge=256)
    RAG_TOP_K: int = Field(default=8, ge=1, le=100)

    # --- Weights & Biases inference tracking ---
    WANDB_ENABLED: bool = Field(default=False)
    WANDB_PROJECT: str = Field(default="vietnamese-rag-gemma")
    WANDB_ENTITY: str | None = Field(default=None)
    WANDB_RUN_NAME: str | None = Field(default=None)
    WANDB_MODE: str = Field(default="online")
    WANDB_LOG_TEXT: bool = Field(default=True)
    WANDB_MAX_TEXT_CHARS: int = Field(default=4000, ge=0)

    # bot persona
    BOT_NAME: str = "VietRAG Bot"
    CREATOR_NAME: str = "NamSyntax"

    # env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


# lru cache (singleton)
# cached settings instance
@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
