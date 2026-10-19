from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AliasChoices, Field, field_validator
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
    # Qwen is the generator for this branch. Keep the legacy environment
    # variable names as aliases for existing notebook configurations.
    HF_MODEL_NAME: str = Field(default="Qwen/Qwen3-4B-Instruct-2507")
    HF_TOKEN: str | None = Field(default=None)
    GENERATOR_DEVICE: str = Field(default="cuda", validation_alias=AliasChoices("GENERATOR_DEVICE", "GEMMA_DEVICE"))
    GENERATOR_MAX_INPUT_TOKENS: int = Field(default=4096, ge=128, validation_alias=AliasChoices("GENERATOR_MAX_INPUT_TOKENS", "GEMMA_MAX_INPUT_TOKENS"))
    LLM_MAX_NEW_TOKENS: int = Field(default=256, ge=1, le=2048)
    RAG_MAX_CONTEXT_CHARS: int = Field(default=12000, ge=256)
    RAG_TOP_K: int = Field(default=8, ge=1, le=100)

    # --- Weights & Biases inference tracking ---
    WANDB_ENABLED: bool = Field(default=False)
    WANDB_PROJECT: str = Field(default="vietnamese-rag-qwen")
    WANDB_ENTITY: str | None = Field(default=None)
    WANDB_RUN_NAME: str | None = Field(default=None)
    WANDB_MODE: str = Field(default="online")
    WANDB_LOG_TEXT: bool = Field(default=True)
    WANDB_MAX_TEXT_CHARS: int = Field(default=4000, ge=0)

    # bot persona
    BOT_NAME: str = "VietRAG Bot"
    CREATOR_NAME: str = "NamSyntax"

    @field_validator("HF_MODEL_NAME")
    @classmethod
    def require_qwen_generator(cls, value: str) -> str:
        if value != "Qwen/Qwen3-4B-Instruct-2507":
            raise ValueError("qwen-rag-generation requires HF_MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507")
        return value

    # env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


# lru cache (singleton)
# cached settings instance
@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
