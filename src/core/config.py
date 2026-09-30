from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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

    # --- LLM config ---
    D2L_CHECKPOINT_PATH: Path = (
        PROJECT_ROOT / "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"
    )
    D2L_MAX_INPUT_TOKENS: int = Field(default=4096, ge=128)
    D2L_MAX_CONTEXT_TOKENS: int = Field(default=2048, ge=32)
    LLM_MAX_NEW_TOKENS: int = Field(default=256, ge=1, le=2048)
    TRUSTMARGIN_LAMBDA_BIND: float = Field(default=0.5, ge=0, allow_inf_nan=False)
    TRUSTMARGIN_TAU: float = Field(default=-1.5, allow_inf_nan=False)

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
