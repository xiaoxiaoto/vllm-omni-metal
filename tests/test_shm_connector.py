# SPDX-License-Identifier: Apache-2.0
import uuid
from multiprocessing import shared_memory
from pathlib import Path

import pytest
import torch

from vllm_omni_metal.shm_connector import MetalSharedMemoryConnector


@pytest.mark.parametrize("with_metadata", [False, True])
def test_long_logical_keys_roundtrip_and_cleanup(with_metadata):
    sender = MetalSharedMemoryConnector({})
    receiver = MetalSharedMemoryConnector({})
    key = "request-" + uuid.uuid4().hex + "_stage0_replica0_chunk0"
    value = {"codes": {"audio": torch.arange(16).reshape(2, 8)}, "finished": True}
    try:
        success, size, metadata = sender.put("0", "1", key, value)
        assert success and size > 0
        assert len(metadata["shm"]["name"].encode()) < 31
        result = receiver.get("0", "1", key, metadata if with_metadata else None)
        assert result is not None
        payload, _ = result
        assert payload["finished"] is True
        torch.testing.assert_close(payload["codes"]["audio"], value["codes"]["audio"])
        assert receiver.get("0", "1", key) is None
        sender.reap_consumed()
        assert not sender._pending_keys
        assert not Path(sender._lock_path(key)).exists()
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=metadata["shm"]["name"])
    finally:
        sender.cleanup(key)
        sender.close()
        receiver.close()


def test_lock_paths_cannot_escape_private_directory():
    connector = MetalSharedMemoryConnector({})
    assert Path(connector._lock_path("../../outside")).parent == connector.lock_dir
    assert connector._physical_key("a") != connector._physical_key("b")
    with pytest.raises(ValueError):
        connector._lock_path("")
