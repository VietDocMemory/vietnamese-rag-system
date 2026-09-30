"""In-process Doc-to-LoRA generation and TrustMargin arbitration."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import uuid

from src.core.config import PROJECT_ROOT, settings
from src.generation.d2l_backend import D2LBackend


class RAGGenerator:
    def __init__(self, backend=None):
        self._backend = backend
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="d2l")
        self._closed = False
        self._revision = uuid.uuid4().hex
        checkpoint = settings.D2L_CHECKPOINT_PATH
        self._checkpoint = checkpoint if checkpoint.is_absolute() else PROJECT_ROOT / checkpoint

    def _load(self):
        # Executed only on our single inference thread; concurrent first calls load once.
        if self._backend is None:
            self._backend = D2LBackend.load(
                self._checkpoint,
                max_input_tokens=settings.D2L_MAX_INPUT_TOKENS,
                max_context_tokens=settings.D2L_MAX_CONTEXT_TOKENS,
            )
        return self._backend

    async def _run(self, function, *args):
        if self._closed:
            raise RuntimeError("Doc-to-LoRA đã đóng.")
        return await asyncio.get_running_loop().run_in_executor(self._executor, function, *args)

    async def initialize(self):
        await self._run(self._load)

    async def aclose(self):
        if self._closed:
            return
        self._closed = True
        # A cancelled HTTP request does not stop running CUDA work. Drain it before
        # releasing the model; queued jobs are cancelled without touching adapters.
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)
        self._backend = None

    def _payload(self, query, contexts, mode):
        return {
            "query": query,
            "contexts": [
                {key: doc.get(key) for key in ("content", "page", "chunk_index")}
                for doc in contexts
            ],
            "mode": mode,
            "lambda_bind": settings.TRUSTMARGIN_LAMBDA_BIND,
            "tau": settings.TRUSTMARGIN_TAU,
            "max_new_tokens": settings.LLM_MAX_NEW_TOKENS,
        }

    async def cache_key(self, query, contexts, mode="auto"):
        await self.initialize()
        fingerprint = {
            "runtime": self._revision,
            "model": self._backend.model_name,
            "payload": self._payload(query, contexts, mode),
            "protocol": "inprocess-d2l-trustmargin-v2",
        }
        return hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    def _infer(self, payload):
        return self._load().infer(**payload)

    async def generate(self, query, contexts, mode="auto"):
        # No HTTP, subprocess or separate model service. The thread keeps FastAPI
        # responsive while the shared model performs blocking PyTorch computation.
        return await self._run(self._infer, self._payload(query, contexts, mode))

    async def generate_stream(self, query, contexts):
        # Compatibility for the offline evaluator. Arbitration must finish first.
        result = await self.generate(query, contexts)
        yield result["answer"]
