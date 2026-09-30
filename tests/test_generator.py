import asyncio
import threading

import pytest

from src.core.config import settings
from src.generation.d2l_backend import D2LBackend
from src.generation.generator import RAGGenerator


class Backend:
    model_name = "checkpoint/model"

    def infer(self, **kwargs):
        self.payload = kwargs
        self.thread_id = threading.get_ident()
        return {
            "answer": "selected answer",
            "sources": kwargs["contexts"],
            "routing": {"selected": "d2l"},
        }


async def test_direct_inference_runs_in_same_process_off_event_loop():
    backend = Backend()
    generator = RAGGenerator(backend)
    try:
        answer = await generator.generate("question", [{"content": "evidence"}])
        assert answer["answer"] == "selected answer"
        assert backend.thread_id != threading.get_ident()
        assert backend.payload["mode"] == "auto"
        assert backend.payload["contexts"][0]["content"] == "evidence"
        assert "model" not in backend.payload
        assert [part async for part in generator.generate_stream("q", [{"content": "p"}])] == [
            "selected answer"
        ]
    finally:
        await generator.aclose()


async def test_concurrent_initialization_loads_once(monkeypatch):
    loads = []
    backend = Backend()

    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return backend

    monkeypatch.setattr(D2LBackend, "load", load)
    generator = RAGGenerator()
    try:
        await asyncio.gather(
            generator.initialize(),
            generator.initialize(),
            generator.generate("q", [{"content": "p"}]),
        )
        assert len(loads) == 1
        assert loads[0][0][0].is_absolute()
    finally:
        await generator.aclose()
    with pytest.raises(RuntimeError, match="đóng"):
        await generator.initialize()


async def test_cache_identity_includes_model_instance_evidence_mode_parameters(monkeypatch):
    generator = RAGGenerator(Backend())
    other = RAGGenerator(Backend())
    docs = [{"content": "original"}]
    try:
        key = await generator.cache_key("Q", docs)
        assert key == await generator.cache_key("Q", docs)
        assert key != await other.cache_key("Q", docs)
        assert key != await generator.cache_key("q", docs)
        assert key != await generator.cache_key("Q", [{"content": "changed"}])
        assert key != await generator.cache_key("Q", docs, "d2l")
        monkeypatch.setattr(settings, "TRUSTMARGIN_TAU", -0.1)
        assert key != await generator.cache_key("Q", docs)
    finally:
        await generator.aclose()
        await other.aclose()


async def test_cancellation_keeps_running_model_owned_until_shutdown():
    started = threading.Event()
    finish = threading.Event()
    released = threading.Event()

    class BlockingBackend(Backend):
        def infer(self, **kwargs):
            started.set()
            try:
                assert finish.wait(timeout=5)
                return super().infer(**kwargs)
            finally:
                released.set()

    generator = RAGGenerator(BlockingBackend())
    task = asyncio.create_task(generator.generate("q", [{"content": "p"}]))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        closing = asyncio.create_task(generator.aclose())
        await asyncio.sleep(0.01)
        assert not closing.done()
        assert not released.is_set()
        finish.set()
        await closing
        assert released.is_set()
        assert generator._backend is None
    finally:
        finish.set()
        await generator.aclose()


async def test_load_failure_is_reported_without_inference_fallback(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("missing checkpoint")

    monkeypatch.setattr(D2LBackend, "load", fail)
    generator = RAGGenerator()
    try:
        with pytest.raises(RuntimeError, match="missing checkpoint"):
            await generator.initialize()
    finally:
        await generator.aclose()
