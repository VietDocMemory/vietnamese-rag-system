"""Benchmark all 420 legal QA requests and measure end-to-end compute resources."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def aggregate_samples(samples: list[dict]) -> dict:
    result = {"sample_count": len(samples)}
    for key in (
        "system_cpu_percent",
        "ram_used_gib",
        "ram_available_gib",
        "gpu_memory_used_mib",
        "gpu_utilization_percent",
        "gpu_power_draw_w",
    ):
        values = [float(row[key]) for row in samples if row.get(key) is not None]
        result[f"{key}_mean"] = statistics.fmean(values) if values else None
        result[f"{key}_p95"] = percentile(values, 0.95)
        result[f"{key}_peak"] = max(values) if values else None
    return result


def query_api(api_url: str, session_id: str, question: str, timeout: float) -> dict:
    started = time.perf_counter()
    sources = []
    answer_parts = []
    error = ""
    sources_at = None
    first_token_at = None
    with requests.get(
        api_url,
        params={"session_id": session_id, "query": question, "bypass_cache": "true"},
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            event = json.loads(raw_line.decode("utf-8"))
            event_type = event.get("type")
            now = time.perf_counter()
            if event_type == "sources":
                sources = event.get("data") or []
                sources_at = sources_at or now
            elif event_type == "content":
                first_token_at = first_token_at or now
                answer_parts.append(str(event.get("data", "")))
            elif event_type == "error":
                error = str(event.get("message", "unknown API error"))
    ended = time.perf_counter()
    return {
        "answer": "".join(answer_parts),
        "sources": sources,
        "error": error,
        "latency_seconds": ended - started,
        "retrieval_to_sources_seconds": sources_at - started if sources_at else None,
        "time_to_first_token_seconds": first_token_at - started if first_token_at else None,
        "generation_after_first_token_seconds": ended - first_token_at if first_token_at else None,
    }


def check_api_ready(api_url: str, timeout: float = 10) -> None:
    parsed = urlsplit(api_url)
    docs_url = urlunsplit((parsed.scheme, parsed.netloc, "/docs", "", ""))
    try:
        response = requests.get(docs_url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"RAG API chưa sẵn sàng tại {docs_url}. "
            "Hãy khởi động uvicorn và kiểm tra api.log trước khi benchmark."
        ) from exc


def build_summary(
    rows: list[dict], inventory: dict, samples: list[dict], session_elapsed: float
) -> dict:
    successful = [row for row in rows if not row.get("error")]
    latencies = [float(row["latency_seconds"]) for row in successful]
    resources = aggregate_samples(samples)
    peak_ram = resources.get("ram_used_gib_peak")
    peak_vram = resources.get("gpu_memory_used_mib_peak")
    cpu_p95 = resources.get("system_cpu_percent_p95")
    logical_cpus = inventory.get("logical_cpu_count") or 1
    measured_request_seconds = sum(latencies)

    has_valid_measurement = bool(successful)
    empirical_recommendation = {
        "cpu_logical_cores": (
            min(
                logical_cpus,
                max(2, math.ceil((cpu_p95 or 100) / 100 * logical_cpus * 1.2)),
            )
            if has_valid_measurement
            else None
        ),
        "ram_gib": (
            math.ceil((peak_ram or 0) * 1.2)
            if has_valid_measurement and peak_ram is not None
            else None
        ),
        "gpu_vram_gib": (
            math.ceil((peak_vram or 0) / 1024 * 1.2)
            if has_valid_measurement and peak_vram is not None
            else None
        ),
        "headroom_factor": 1.2,
        "warning": (
            "Đây là cấu hình ước lượng từ tải tuần tự 1 request trên máy benchmark, "
            "không phải yêu cầu tuyệt đối. Hãy benchmark lại khi đổi model, top-k hoặc concurrency."
        ),
    }
    return {
        "question_count": len(rows),
        "successful_count": len(successful),
        "error_count": len(rows) - len(successful),
        "session_elapsed_seconds": session_elapsed,
        "measured_request_seconds": measured_request_seconds,
        "throughput_questions_per_minute": (
            len(successful) / measured_request_seconds * 60 if measured_request_seconds else None
        ),
        "latency_seconds": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else None,
        },
        "resources": resources,
        "machine_inventory": inventory,
        "empirical_minimum_recommendation": empirical_recommendation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/rag_evaluation_dataset.jsonl"))
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/ask")
    parser.add_argument("--session-id", default="full_legal_109_pages")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/resource_benchmark"))
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    from src.evaluation.resource_monitor import ResourceMonitor, machine_inventory

    args = parse_args()
    check_api_ready(args.api_url)
    dataset = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit > 0:
        dataset = dataset[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "per_question.jsonl"
    samples_path = args.output_dir / "resource_samples.jsonl"
    summary_path = args.output_dir / "summary.json"
    if args.overwrite:
        for path in (results_path, samples_path, summary_path):
            path.unlink(missing_ok=True)

    finished = set()
    rows = []
    if results_path.exists():
        rows = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        finished = {int(row["index"]) for row in rows}

    previous_samples = []
    if samples_path.exists():
        previous_samples = [
            json.loads(line)
            for line in samples_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    inventory = machine_inventory(Path.cwd())
    benchmark_started = time.perf_counter()
    print("=== RAG resource benchmark ===", flush=True)
    print("Tổng số câu:", len(dataset), flush=True)
    print("Đã hoàn thành:", len(finished), flush=True)
    print("GPU:", [gpu["name"] for gpu in inventory["gpus"]] or "không phát hiện", flush=True)

    with ResourceMonitor(args.sample_interval) as monitor:
        with results_path.open("a", encoding="utf-8") as output_file:
            for index, item in enumerate(dataset, start=1):
                if index in finished:
                    continue
                sample_start = monitor.mark()
                try:
                    response = query_api(
                        args.api_url,
                        args.session_id,
                        item["question"],
                        args.timeout,
                    )
                except Exception as exc:  # keep the 420-row run resumable
                    response = {
                        "answer": "",
                        "sources": [],
                        "error": f"{type(exc).__name__}: {exc}",
                        "latency_seconds": 0.0,
                        "retrieval_to_sources_seconds": None,
                        "time_to_first_token_seconds": None,
                        "generation_after_first_token_seconds": None,
                    }
                request_samples = monitor.rows_since(sample_start)
                row = {
                    "index": index,
                    "question_type": item.get("question_type"),
                    "question": item["question"],
                    "ground_truth_answer": item.get("ground_truth_answer"),
                    **response,
                    "resources": aggregate_samples(request_samples),
                }
                rows.append(row)
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_file.flush()
                status = "LỖI" if row["error"] else "xong"
                print(
                    f"[{index}/{len(dataset)}] {status} | {row['latency_seconds']:.2f}s | "
                    f"GPU peak={row['resources'].get('gpu_memory_used_mib_peak')} MiB",
                    flush=True,
                )
        monitor.stop()
        all_samples = previous_samples + monitor.rows_since(0)

    samples_path.write_text(
        "".join(json.dumps(sample) + "\n" for sample in all_samples),
        encoding="utf-8",
    )
    session_elapsed = time.perf_counter() - benchmark_started
    summary = build_summary(rows, inventory, all_samples, session_elapsed)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    recommendation = summary["empirical_minimum_recommendation"]
    print("\n=== Kết quả tài nguyên ===")
    print("Câu thành công:", summary["successful_count"], "/", summary["question_count"])
    print("Latency P50:", summary["latency_seconds"]["p50"])
    print("Latency P95:", summary["latency_seconds"]["p95"])
    print("Peak RAM GiB:", summary["resources"]["ram_used_gib_peak"])
    print("Peak VRAM MiB:", summary["resources"]["gpu_memory_used_mib_peak"])
    print("Ước lượng CPU logical cores:", recommendation["cpu_logical_cores"])
    print("Ước lượng RAM GiB:", recommendation["ram_gib"])
    print("Ước lượng VRAM GiB:", recommendation["gpu_vram_gib"])
    print("\nSummary:", summary_path)
    print("Từng câu:", results_path)
    print("Raw samples:", samples_path)


if __name__ == "__main__":
    main()
