from unittest.mock import AsyncMock, MagicMock

import pytest

import src.api.main as main


async def test_api_initializes_and_closes_embedded_generator(monkeypatch):
    generator = MagicMock(initialize=AsyncMock(), aclose=AsyncMock())
    monkeypatch.setattr(main, "RAGGenerator", lambda: generator)
    for name in ("RAGRetriever", "VectorStoreManager", "PDFIngestionPipeline"):
        monkeypatch.setattr(main, name, MagicMock())
    async with main.lifespan(main.app):
        generator.initialize.assert_awaited_once()
        assert main.app.state.generator is generator
    generator.aclose.assert_awaited_once()


async def test_failed_model_startup_cleans_up_and_does_not_claim_readiness(monkeypatch):
    generator = MagicMock(
        initialize=AsyncMock(side_effect=RuntimeError("no checkpoint")), aclose=AsyncMock()
    )
    monkeypatch.setattr(main, "RAGGenerator", lambda: generator)
    with pytest.raises(RuntimeError, match="no checkpoint"):
        async with main.lifespan(main.app):
            pytest.fail("Startup must fail")
    generator.aclose.assert_awaited_once()
