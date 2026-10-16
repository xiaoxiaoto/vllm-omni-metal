# SPDX-License-Identifier: Apache-2.0
"""Inline diffusion workers must release pipeline-owned operator threads."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from vllm_omni_metal.worker import MetalDiffusionWorker


@pytest.mark.parametrize("fail_after_close", [False, True])
def test_shutdown_joins_operator_threads_and_releases_runner(fail_after_close):
    class Pipeline:
        def __init__(self):
            self.pool = ThreadPoolExecutor(max_workers=1)
            self.result = self.pool.submit(lambda: 42)
            self.closes = 0

        def close_metal_operators(self):
            self.closes += 1
            self.pool.shutdown(wait=True)
            if fail_after_close:
                raise RuntimeError("operator release failed")

    pipeline = Pipeline()
    worker = object.__new__(MetalDiffusionWorker)
    worker.model_runner = SimpleNamespace(pipeline=pipeline)
    assert not getattr(worker, "_shutdown_complete", False)
    if fail_after_close:
        with pytest.raises(RuntimeError, match="operator release failed"):
            worker.shutdown()
    else:
        worker.shutdown()
    assert pipeline.result.done()
    assert worker.model_runner is None
    assert worker._shutdown_complete
    assert not any(thread.is_alive() for thread in pipeline.pool._threads)
    worker.shutdown()
    assert pipeline.closes == 1
