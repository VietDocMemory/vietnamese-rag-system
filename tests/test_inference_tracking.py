from src.core.inference_tracking import WandbInferenceTracker


class FakeRun:
    def __init__(self):
        self.payloads = []

    def log(self, payload):
        self.payloads.append(payload)


class FakeWandb:
    @staticmethod
    def Table(columns, data):
        return {"columns": columns, "data": data}


async def test_disabled_tracker_does_not_import_wandb(monkeypatch):
    tracker = WandbInferenceTracker()
    tracker.enabled = False

    def fail_if_called(_record):
        raise AssertionError("Disabled tracking must not start a W&B worker")

    monkeypatch.setattr(tracker, "_log_sync", fail_if_called)

    await tracker.log({"question": "test"})


def test_tracker_logs_metrics_and_sample_table(monkeypatch):
    tracker = WandbInferenceTracker()
    fake_run = FakeRun()
    tracker._wandb = FakeWandb()
    monkeypatch.setattr(tracker, "_ensure_run", lambda: fake_run)

    tracker._log_sync(
        {
            "request_id": "req-1",
            "session_id": "session-1",
            "question": "Câu hỏi",
            "answer": "Câu trả lời",
            "contexts": [{"page": 1, "chunk_index": 2, "content": "Bằng chứng"}],
            "model_metrics": {"prompt_tokens": 20, "output_tokens": 5, "total_tokens": 25},
            "retrieval_seconds": 0.1,
            "generation_seconds": 0.2,
            "end_to_end_seconds": 0.3,
        }
    )

    payload = fake_run.payloads[0]
    assert payload["tokens/total"] == 25
    assert payload["inference/context_count"] == 1
    assert payload["latency/end_to_end_seconds"] == 0.3
    assert "inference/sample" in payload
