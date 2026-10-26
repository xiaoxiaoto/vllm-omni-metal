# SPDX-License-Identifier: Apache-2.0
"""Pure measured allocation behavior; no graph, device, or timing mocks."""

import math

import pytest

from vllm_omni_metal.ane.allocation import RowAllocator


def observe(allocator, rows=1025, chunks=1, gpu=769.0, ane=256.0):
    return allocator.observe(total_rows=rows, ane_chunks=chunks, gpu_seconds=gpu, ane_seconds=ane)


def test_measured_balanced_prediction():
    allocator = RowAllocator()
    assert observe(allocator) == 2
    assert allocator.gpu_seconds_per_row == 1
    assert allocator.ane_seconds_per_chunk == 256
    assert allocator.prediction(1025, 2) == 513
    assert allocator.prediction(1025, 0) == 1025
    assert allocator.prediction(1025, 3) == 768


def test_actual_current_split_not_previous_selection_controls_threshold():
    allocator = RowAllocator(initial_chunks=3)
    assert observe(allocator) == 2
    # An explicitly measured all-GPU run is compared against all-GPU, not
    # against the prior selected split (whose prediction is already optimal).
    assert observe(allocator, chunks=0, gpu=1025, ane=0) == 2


def test_rate_updates_use_half_weighted_ema():
    allocator = RowAllocator()
    observe(allocator)
    observe(allocator, gpu=769 * 3, ane=256 * 5)
    assert allocator.gpu_seconds_per_row == 2
    assert allocator.ane_seconds_per_chunk == 768
    observe(allocator, chunks=0, gpu=1025 * 4, ane=0)
    assert allocator.gpu_seconds_per_row == 3
    assert allocator.ane_seconds_per_chunk == 768


@pytest.mark.parametrize("rows,expected", [(39, 1), (40, 0), (41, 0)])
def test_change_requires_strictly_more_than_five_percent(rows, expected):
    allocator = RowAllocator(chunk_rows=2)
    observe(allocator, rows=40, chunks=1, gpu=38, ane=36)
    assert observe(allocator, rows=rows, chunks=0, gpu=rows, ane=0) == expected


def test_prediction_ties_preserve_current_split():
    allocator = RowAllocator(chunk_rows=1)
    assert observe(allocator, rows=5, chunks=2, gpu=3, ane=2) == 2
    assert allocator.prediction(5, 2) == allocator.prediction(5, 3) == 3
    assert observe(allocator, rows=5, chunks=3, gpu=2, ane=3) == 3


def test_enumeration_matches_exhaustive_formula():
    for rows in (257, 513, 769, 1025, 2049):
        for gpu_rate, ane_rate in ((0.1, 100), (1, 256), (3, 80)):
            allocator = RowAllocator()
            current = (rows - 1) // 256
            actual = observe(
                allocator,
                rows=rows,
                chunks=current,
                gpu=gpu_rate * (rows - current * 256),
                ane=ane_rate * current,
            )
            times = {
                chunks: max(gpu_rate * (rows - chunks * 256), ane_rate * chunks)
                for chunks in range(1, current + 1)
            }
            best = min(times.values())
            if best < times[current] * 0.95:
                assert times[actual] == best
            else:
                assert actual == current
            assert rows - actual * 256 >= 1


def test_gpu_only_initial_state_has_no_fabricated_ane_rate():
    allocator = RowAllocator(initial_chunks=0)
    assert observe(allocator, chunks=0, gpu=1025, ane=0) == 0
    assert allocator.gpu_seconds_per_row == 1
    assert allocator.ane_seconds_per_chunk is None
    with pytest.raises(RuntimeError, match="measured GPU and ANE"):
        allocator.prediction(1025, 0)
    assert observe(allocator) == 2


@pytest.mark.parametrize("rows", [1, 19, 255, 256])
def test_short_input_is_all_gpu_without_rate_pollution(rows):
    allocator = RowAllocator()
    observe(allocator)
    before = allocator.gpu_seconds_per_row, allocator.ane_seconds_per_chunk
    assert observe(allocator, rows=rows, chunks=0, gpu=999, ane=0) == 0
    assert allocator.selected_chunks == 0
    assert (allocator.gpu_seconds_per_row, allocator.ane_seconds_per_chunk) == before
    assert observe(allocator) == 2


def test_smallest_legal_mixed_split_retains_one_real_gpu_row():
    allocator = RowAllocator()
    assert observe(allocator, rows=257, chunks=1, gpu=1, ane=1) == 1
    assert allocator.prediction(257, 1) == 1
    with pytest.raises(ValueError, match="retain a GPU row"):
        allocator.prediction(256, 1)


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "256", None])
def test_invalid_chunk_rows_are_rejected(value):
    with pytest.raises(ValueError, match="chunk_rows"):
        RowAllocator(chunk_rows=value)


@pytest.mark.parametrize("value", [True, False, -1, 1.0, "1", None])
def test_invalid_initial_chunks_are_rejected(value):
    with pytest.raises(ValueError, match="initial_chunks"):
        RowAllocator(initial_chunks=value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("total_rows", True),
        ("total_rows", 0),
        ("total_rows", -1),
        ("total_rows", 1025.0),
        ("ane_chunks", True),
        ("ane_chunks", -1),
        ("ane_chunks", 1.0),
        ("ane_chunks", 5),
        ("gpu_seconds", True),
        ("gpu_seconds", 0),
        ("gpu_seconds", -1),
        ("gpu_seconds", math.inf),
        ("gpu_seconds", math.nan),
        ("ane_seconds", False),
        ("ane_seconds", 0),
        ("ane_seconds", -1),
        ("ane_seconds", math.inf),
        ("ane_seconds", math.nan),
        ("ane_seconds", "1"),
    ],
)
def test_invalid_observation_does_not_partially_update_rates(field, value):
    allocator = RowAllocator()
    observe(allocator)
    before = (
        allocator.selected_chunks,
        allocator.gpu_seconds_per_row,
        allocator.ane_seconds_per_chunk,
    )
    sample = dict(total_rows=1025, ane_chunks=1, gpu_seconds=1538, ane_seconds=512)
    sample[field] = value
    with pytest.raises(ValueError):
        allocator.observe(**sample)
    assert (
        allocator.selected_chunks,
        allocator.gpu_seconds_per_row,
        allocator.ane_seconds_per_chunk,
    ) == before


def test_unexecuted_ane_requires_zero_duration():
    with pytest.raises(ValueError, match="unexecuted ANE"):
        observe(RowAllocator(), chunks=0, gpu=1025, ane=1)


@pytest.mark.parametrize("rows,chunks", [(True, 0), (0, 0), (257, True), (257, 2)])
def test_prediction_validates_rows_and_chunks_before_use(rows, chunks):
    allocator = RowAllocator()
    with pytest.raises(ValueError):
        allocator.prediction(rows, chunks)


def test_unmeasured_prediction_is_rejected():
    with pytest.raises(RuntimeError, match="measured GPU and ANE"):
        RowAllocator().prediction(513, 1)
