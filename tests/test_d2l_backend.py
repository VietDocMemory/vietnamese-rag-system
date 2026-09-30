from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import time

import pytest
import torch

from src.generation.d2l_backend import D2LBackend, mean_answer_logprob, prompt_views


def test_teacher_forcing_masks_prompt_and_shifts_answer():
    logits = torch.zeros(1, 5, 7)
    logits[0, 2, 4] = 2  # first answer token predicted by final prompt position
    logits[0, 3, 5] = 3
    logits[0, 0, :] = 90  # prompt loss must not contribute

    def model(**kwargs):
        assert kwargs["input_ids"].tolist() == [[1, 2, 3, 4, 5]]
        assert kwargs["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]
        assert kwargs["use_cache"] is False
        return SimpleNamespace(logits=logits)

    score = mean_answer_logprob(model, torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]]))
    expected = (torch.log_softmax(logits[0, 2], -1)[4] + torch.log_softmax(logits[0, 3], -1)[5]) / 2
    assert score == pytest.approx(expected.item())


class Tokenizer:
    all_special_ids = [0]
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return torch.ones((1, len(messages[0]["content"]) + 2), dtype=torch.long)

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


class Model:
    device = "cpu"

    def __init__(self):
        self.adapted = False
        self.events = []

    @property
    def base_model(self):
        return self

    def reset(self):
        self.adapted = False
        self.events.append("reset")

    def _internalize_from_ids(self, context_ids):
        assert context_ids.ndim == 2
        self.adapted = True
        self.events.append("internalize")
        time.sleep(0.005)

    def generate(self, input_ids, **kwargs):
        self.events.append("d2l" if self.adapted else "rag")
        answer = torch.tensor([[3 if self.adapted else 2, 0]])
        return torch.cat((input_ids, answer), dim=1)

    def __call__(self, input_ids, **kwargs):
        assert self.adapted, "All six scores must keep the D2L memory state"
        self.events.append("score")
        return SimpleNamespace(logits=torch.zeros(1, input_ids.shape[1], 6))


def backend(model=None, **kwargs):
    return D2LBackend(
        model or Model(),
        Tokenizer(),
        Tokenizer(),
        lambda data, tok: {"ctx_ids": [[1] * (len(data["context"][0]) + 4)]},
        "checkpoint/model",
        **kwargs,
    )


def test_two_candidates_six_scores_and_reset():
    engine = backend()
    result = engine.infer("question?", [{"content": "evidence"}])
    assert engine.model.events == ["reset", "rag", "internalize", "d2l"] + ["score"] * 6 + ["reset"]
    assert result["routing"]["selected"] == "rag"
    assert result["routing"]["model"] == "checkpoint/model"
    assert result["answer"] == "2"
    assert not engine.model.adapted


@pytest.mark.parametrize("mode", ["rag", "d2l"])
def test_forced_mode_skips_other_branch_and_scoring(mode):
    engine = backend()
    result = engine.infer("question", [{"content": "evidence"}], mode=mode)
    assert result["routing"]["selected"] == mode
    assert "score" not in engine.model.events
    assert ("d2l" if mode == "rag" else "rag") not in engine.model.events


def test_reset_on_failure_and_next_request_is_clean(monkeypatch):
    engine = backend()
    original = engine._generate

    def fail(prefix, count):
        if engine.model.adapted:
            raise RuntimeError("simulated generation failure")
        return original(prefix, count)

    monkeypatch.setattr(engine, "_generate", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        engine.infer("question", [{"content": "evidence"}])
    assert not engine.model.adapted
    monkeypatch.setattr(engine, "_generate", original)
    assert engine.infer("next", [{"content": "other"}], mode="rag")["answer"] == "2"


def test_concurrent_requests_are_serialized():
    engine = backend()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda q: engine.infer(q, [{"content": q}], mode="d2l"), ["first", "second"])
        )
    assert len(results) == 2
    assert engine.model.events == ["reset", "internalize", "d2l", "reset"] * 2


def test_context_budget_applies_to_encoder_and_returned_sources():
    engine = backend(max_context_tokens=50)
    result = engine.infer("q", [{"content": "a" * 100, "page": 7}], mode="rag")
    assert result["routing"]["context_truncated"]
    assert 0 < len(result["sources"][0]["content"]) < 100
    assert result["sources"][0]["page"] == 7


def test_overlong_query_rejected_and_reset():
    engine = backend(max_input_tokens=100)
    with pytest.raises(ValueError, match="token"):
        engine.infer("q" * 150, [{"content": "evidence"}])
    assert engine.model.events == ["reset", "reset"]


def test_context_only_view_omits_question():
    views = prompt_views("UNIQUE_QUESTION", "SHARED_EVIDENCE")
    assert "UNIQUE_QUESTION" not in views["context"]
    assert "SHARED_EVIDENCE" not in views["question"]
    assert "SHARED_EVIDENCE" in views["question_context"]
    assert "SHARED_EVIDENCE" in views["context"]
