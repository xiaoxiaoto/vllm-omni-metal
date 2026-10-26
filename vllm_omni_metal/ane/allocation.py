# SPDX-License-Identifier: Apache-2.0
"""Measured offline GPU/ANE allocation; no compilation or approval decisions.

Predictions use synchronized per-route durations, not whole-pipeline elapsed
samples. Allocation does not replace numerical approval or the independent 3%
whole-pipeline publication gate. No input is padded or repeated.
"""

import math

from vllm_omni_metal.ane.cache import CHUNK


class RowAllocator:
    """Predict mixed-route time with measured rate EMA alpha=0.5.

    ``T(c) = max(gpu_seconds_per_row * (M-c*C), ane_seconds_per_chunk*c)``.
    Enumerate complete ANE chunks while retaining a real GPU row; change the
    measured allocation only for a predicted improvement strictly above 5%.
    Callers own device synchronization, graph preparation, and freezing the
    chosen allocation before any production replay.
    """

    def __init__(self, chunk_rows=CHUNK, initial_chunks=1):
        if type(chunk_rows) is not int or chunk_rows <= 0:
            raise ValueError("chunk_rows must be a positive integer")
        if type(initial_chunks) is not int or initial_chunks < 0:
            raise ValueError("initial_chunks must be a nonnegative integer")
        self.chunk_rows = chunk_rows
        self._selected_chunks = initial_chunks
        self._gpu_seconds_per_row = None
        self._ane_seconds_per_chunk = None

    @property
    def selected_chunks(self):
        return self._selected_chunks

    @property
    def gpu_seconds_per_row(self):
        return self._gpu_seconds_per_row

    @property
    def ane_seconds_per_chunk(self):
        return self._ane_seconds_per_chunk

    def _split(self, total_rows, chunks):
        if type(total_rows) is not int or total_rows < 1:
            raise ValueError("total_rows must be a positive integer")
        if type(chunks) is not int or not 0 <= chunks <= (total_rows - 1) // self.chunk_rows:
            raise ValueError("ANE chunks must be nonnegative and retain a GPU row")
        return total_rows - chunks * self.chunk_rows

    def prediction(self, total_rows, chunks):
        """Return predicted seconds; missing measured rates are an explicit error."""
        gpu_rows = self._split(total_rows, chunks)
        if self._gpu_seconds_per_row is None or self._ane_seconds_per_chunk is None:
            raise RuntimeError("prediction requires measured GPU and ANE rates")
        return max(
            self._gpu_seconds_per_row * gpu_rows,
            self._ane_seconds_per_chunk * chunks,
        )

    def observe(self, *, total_rows, ane_chunks, gpu_seconds, ane_seconds):
        """Update measured rates atomically and return the selected chunk count.

        The supplied ``ane_chunks`` is the current, actually measured split.
        All-GPU observations use ``ane_chunks=0, ane_seconds=0``. Inputs too
        short for mixed execution return zero without changing either rate.
        Executed routes require positive finite measured durations in seconds.
        """
        gpu_rows = self._split(total_rows, ane_chunks)
        for duration in (gpu_seconds, ane_seconds):
            if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
                raise ValueError("route durations must be finite nonnegative seconds")
        if gpu_seconds <= 0 or (ane_chunks and ane_seconds <= 0):
            raise ValueError("executed routes require positive measured seconds")
        if not ane_chunks and ane_seconds != 0:
            raise ValueError("an unexecuted ANE route must have zero seconds")
        if total_rows <= self.chunk_rows:
            self._selected_chunks = 0
            return 0
        gpu_rate = gpu_seconds / gpu_rows
        ane_rate = ane_seconds / ane_chunks if ane_chunks else None
        if gpu_rate <= 0 or (ane_rate is not None and ane_rate <= 0):
            raise ValueError("measured route rates must be positive")
        self._gpu_seconds_per_row = (
            gpu_rate
            if self._gpu_seconds_per_row is None
            else 0.5 * self._gpu_seconds_per_row + 0.5 * gpu_rate
        )
        if ane_rate is not None:
            self._ane_seconds_per_chunk = (
                ane_rate
                if self._ane_seconds_per_chunk is None
                else 0.5 * self._ane_seconds_per_chunk + 0.5 * ane_rate
            )
        self._selected_chunks = ane_chunks
        if self._ane_seconds_per_chunk is None:
            return self._selected_chunks
        current_time = self.prediction(total_rows, ane_chunks)
        best_chunks, best_time = ane_chunks, current_time
        for chunks in range(1, (total_rows - 1) // self.chunk_rows + 1):
            elapsed = self.prediction(total_rows, chunks)
            if elapsed < best_time:
                best_chunks, best_time = chunks, elapsed
        if best_time < current_time * 0.95:
            self._selected_chunks = best_chunks
        return self._selected_chunks
