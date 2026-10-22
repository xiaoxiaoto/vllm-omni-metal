# SPDX-License-Identifier: Apache-2.0
"""POSIX shared memory with macOS-compatible, private lock files."""

import fcntl
import hashlib
import os
import tempfile
from multiprocessing import shared_memory
from pathlib import Path

from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.entrypoints.stage_utils import shm_write_bytes


class MetalSharedMemoryConnector(SharedMemoryConnector):
    def __init__(self, config):
        super().__init__(config)
        self.lock_dir = Path(tempfile.gettempdir()) / f"vllm-omni-metal-shm-{os.getuid()}"
        self.lock_dir.mkdir(mode=0o700, exist_ok=True)
        info = self.lock_dir.lstat()
        if self.lock_dir.is_symlink() or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError(
                "shared-memory lock directory must be private and owned by this user"
            )

    @staticmethod
    def _physical_key(key):
        if not isinstance(key, str) or not key or "\x00" in key:
            raise ValueError("shared-memory key must be a nonempty string")
        # Darwin POSIX SHM names are limited to 31 bytes, including the slash.
        digest = hashlib.sha256(f"{os.getuid()}:{key}".encode()).hexdigest()
        return "mps_" + digest[:24]

    def _lock_path(self, key):
        return str(self.lock_dir / (self._physical_key(key) + ".lock"))

    def put(self, from_stage, to_stage, put_key, data):
        payload = self.serialize_obj(data)
        lock_path = self._lock_path(put_key)
        self.reap_consumed()
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                framed = len(payload).to_bytes(8, "little") + payload
                meta = shm_write_bytes(framed, name=self._physical_key(put_key))
            finally:
                # The resource tracker can inherit a descriptor on its first spawn.
                fcntl.flock(lock, fcntl.LOCK_UN)
        with self._pending_keys_lock:
            self._pending_keys[put_key] = None
        self._metrics["puts"] += 1
        self._metrics["bytes_transferred"] += len(payload)
        return True, len(payload), {"shm": meta, "size": len(payload)}

    def _get_data_with_lock(self, lock_file, shm_handle):
        segment = None
        consumed = False
        try:
            fd = os.open(lock_file, os.O_RDWR | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                segment = shared_memory.SharedMemory(name=shm_handle["name"])
                if segment.size < 8:
                    return None
                length = int.from_bytes(segment.buf[:8], "little")
                if length <= 0:
                    return None
                if length > segment.size - 8:
                    raise ValueError("shared-memory frame exceeds allocated segment")
                payload = bytes(segment.buf[8 : 8 + length])
                obj = self.deserialize_obj(payload)
                segment.unlink()
                consumed = True
                return obj, length
        except (BlockingIOError, FileNotFoundError):
            return None
        finally:
            if segment is not None:
                segment.close()
            if consumed:
                try:
                    os.unlink(lock_file)
                except FileNotFoundError:
                    pass

    def _get_by_key(self, get_key):
        try:
            segment = shared_memory.SharedMemory(name=self._physical_key(get_key))
        except FileNotFoundError:
            return None
        except ValueError as error:
            if "empty file" in str(error):
                return None
            raise
        try:
            if segment.size == 0:
                return None
            result = self._get_data_with_lock(
                self._lock_path(get_key),
                {"name": self._physical_key(get_key), "size": segment.size},
            )
            if result is not None:
                with self._pending_keys_lock:
                    self._pending_keys.pop(get_key, None)
            return result
        finally:
            segment.close()

    def get(self, from_stage, to_stage, get_key, metadata=None):
        if isinstance(metadata, dict) and get_key in metadata:
            metadata = metadata[get_key]
        if isinstance(metadata, dict) and "shm" in metadata:
            handle = metadata["shm"]
            if handle["name"] != self._physical_key(get_key):
                raise ValueError("shared-memory metadata does not match logical key")
            result = self._get_data_with_lock(self._lock_path(get_key), handle)
            if result is not None:
                with self._pending_keys_lock:
                    self._pending_keys.pop(get_key, None)
        else:
            result = self._get_by_key(get_key)
        if result is not None:
            self._metrics["gets"] += 1
        return result

    def cleanup(self, request_id):
        with self._pending_keys_lock:
            self._pending_keys.pop(request_id, None)
        try:
            segment = shared_memory.SharedMemory(name=self._physical_key(request_id))
        except FileNotFoundError:
            pass
        else:
            segment.close()
            segment.unlink()
        try:
            os.unlink(self._lock_path(request_id))
        except FileNotFoundError:
            pass

    def reap_consumed(self):
        with self._pending_keys_lock:
            keys = list(self._pending_keys)[:64]
        for key in keys:
            try:
                segment = shared_memory.SharedMemory(name=self._physical_key(key))
            except FileNotFoundError:
                with self._pending_keys_lock:
                    self._pending_keys.pop(key, None)
            else:
                segment.close()


def register_metal_shm_connector():
    from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory

    OmniConnectorFactory._registry["SharedMemoryConnector"] = MetalSharedMemoryConnector
