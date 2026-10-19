"""Optional Weights & Biases logging for online RAG inference."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

import torch

from src.core.config import settings

logger = logging.getLogger(__name__)


class WandbInferenceTracker:
    """Own one W&B run and append one inference record per request.

    Import and initialization are lazy, so W&B is completely inert when
    ``WANDB_ENABLED`` is false. Tracking failures are isolated from the RAG API.
    """

    def __init__(self) -> None:
        self.enabled = settings.WANDB_ENABLED
        self._run: Any | None = None
        self._wandb: Any | None = None
        self._lock = threading.Lock()

    def _ensure_run(self) -> Any:
        if self._run is not None:
            return self._run

        import wandb

        mode = settings.WANDB_MODE.lower()
        if mode not in {"online", "offline", "disabled"}:
            raise ValueError("WANDB_MODE phải là online, offline hoặc disabled.")

        self._wandb = wandb
        self._run = wandb.init(
            project=settings.WANDB_PROJECT,
            entity=settings.WANDB_ENTITY or None,
            name=settings.WANDB_RUN_NAME or None,
            job_type="inference",
            mode=mode,
            config={
                "generator_model": settings.HF_MODEL_NAME,
                "generator_device": settings.GENERATOR_DEVICE,
                "max_input_tokens": settings.GENERATOR_MAX_INPUT_TOKENS,
                "max_new_tokens": settings.LLM_MAX_NEW_TOKENS,
                "max_context_chars": settings.RAG_MAX_CONTEXT_CHARS,
                "retrieval_top_k": settings.RAG_TOP_K,
                "prompt_version": "qwen-rag-v1",
            },
        )
        return self._run

    @staticmethod
    def _gpu_metrics() -> dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        return {
            "system/gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "system/gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "system/gpu_max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        }

    def _truncate(self, value: str) -> str:
        if not settings.WANDB_LOG_TEXT:
            return ""
        limit = settings.WANDB_MAX_TEXT_CHARS
        return value[:limit] if limit else ""

    def _log_sync(self, record: dict[str, Any]) -> None:
        with self._lock:
            run = self._ensure_run()
            contexts = record.get("contexts", [])
            answer = str(record.get("answer", ""))
            model_metrics = record.get("model_metrics", {})
            payload: dict[str, Any] = {
                "inference/cache_hit": int(bool(record.get("cache_hit"))),
                "inference/success": int(bool(record.get("success", True))),
                "inference/context_count": len(contexts),
                "inference/context_chars": sum(len(str(item.get("content", ""))) for item in contexts),
                "inference/answer_chars": len(answer),
                "tokens/prompt": int(model_metrics.get("prompt_tokens", 0)),
                "tokens/output": int(model_metrics.get("output_tokens", 0)),
                "tokens/total": int(model_metrics.get("total_tokens", 0)),
                "latency/retrieval_seconds": float(record.get("retrieval_seconds", 0.0)),
                "latency/generation_seconds": float(record.get("generation_seconds", 0.0)),
                "latency/end_to_end_seconds": float(record.get("end_to_end_seconds", 0.0)),
                **self._gpu_metrics(),
            }

            if settings.WANDB_LOG_TEXT:
                source_rows = [
                    {
                        "page": item.get("page"),
                        "chunk_index": item.get("chunk_index"),
                        "content": self._truncate(str(item.get("content", ""))),
                    }
                    for item in contexts
                ]
                table = self._wandb.Table(
                    columns=[
                        "request_id",
                        "session_id",
                        "question",
                        "answer",
                        "sources_json",
                        "cache_hit",
                        "error",
                    ],
                    data=[
                        [
                            record.get("request_id", ""),
                            record.get("session_id", ""),
                            self._truncate(str(record.get("question", ""))),
                            self._truncate(answer),
                            json.dumps(source_rows, ensure_ascii=False),
                            bool(record.get("cache_hit")),
                            self._truncate(str(record.get("error", ""))),
                        ]
                    ],
                )
                payload["inference/sample"] = table

            run.log(payload)

    async def log(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            await asyncio.to_thread(self._log_sync, record)
        except Exception:
            logger.error("Không thể log inference lên W&B.", exc_info=True)

    def _finish_sync(self) -> None:
        with self._lock:
            if self._run is not None:
                self._run.finish()
                self._run = None

    async def finish(self) -> None:
        if not self.enabled:
            return
        try:
            await asyncio.to_thread(self._finish_sync)
        except Exception:
            logger.error("Không thể đóng W&B run.", exc_info=True)


inference_tracker = WandbInferenceTracker()
