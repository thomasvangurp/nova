from nanobody_opt.compute_budget import TimingModel


def _rounds(gpu_count, per_batch=4.0, load=4.0):
    # 32 candidates on gpu_count GPUs: ceil(32/gpu_count) batches per GPU.
    batches = -(-32 // gpu_count)
    return [{"phase": "replication_round", "candidates": 32, "gpu_count": gpu_count,
             "seconds": load + per_batch * batches} for _ in range(25)]


def test_fit_uses_the_gpu_count_each_round_ran_on():
    """4-GPU seed epochs must not double the per-candidate cost on an 8-GPU box."""
    base = TimingModel(4.0, 4.0, 120)
    on_four = base.fit_from_measurements(_rounds(4), 4)
    four_seen_by_eight = base.fit_from_measurements(_rounds(4), 8)
    assert abs(four_seen_by_eight.gpu_seconds_per_candidate
               - on_four.gpu_seconds_per_candidate) < 1e-9


def test_widen_uses_the_gpu_count_each_round_ran_on():
    base = TimingModel(4.0, 4.0, 120)
    assert abs(base.widen_for_measurements(_rounds(4), 8).gpu_seconds_per_candidate
               - base.widen_for_measurements(_rounds(4), 4).gpu_seconds_per_candidate) < 1e-9


def test_records_without_gpu_count_fall_back_to_current_count():
    rows = [{k: v for k, v in row.items() if k != "gpu_count"} for row in _rounds(8)]
    base = TimingModel(4.0, 4.0, 120)
    assert base.fit_from_measurements(rows, 8) == base.fit_from_measurements(_rounds(8), 8)
