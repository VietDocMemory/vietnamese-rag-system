from scripts.benchmark_rag_resources import aggregate_samples, build_summary, percentile


def test_percentile_interpolates():
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([], 0.95) is None


def test_resource_summary_and_recommendation():
    samples = [
        {
            "system_cpu_percent": 25,
            "ram_used_gib": 10,
            "ram_available_gib": 20,
            "gpu_memory_used_mib": 4096,
            "gpu_utilization_percent": 50,
            "gpu_power_draw_w": 100,
        },
        {
            "system_cpu_percent": 50,
            "ram_used_gib": 12,
            "ram_available_gib": 18,
            "gpu_memory_used_mib": 5120,
            "gpu_utilization_percent": 75,
            "gpu_power_draw_w": 150,
        },
    ]
    aggregate = aggregate_samples(samples)
    assert aggregate["ram_used_gib_peak"] == 12
    assert aggregate["gpu_memory_used_mib_peak"] == 5120

    rows = [{"error": "", "latency_seconds": 2.0}, {"error": "", "latency_seconds": 4.0}]
    summary = build_summary(
        rows,
        {"logical_cpu_count": 8},
        samples,
        session_elapsed=8,
    )
    assert summary["successful_count"] == 2
    assert summary["latency_seconds"]["p50"] == 3.0
    assert summary["empirical_minimum_recommendation"]["ram_gib"] == 15
    assert summary["empirical_minimum_recommendation"]["gpu_vram_gib"] == 6
