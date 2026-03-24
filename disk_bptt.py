"""
disk_bptt — Full-sequence backpropagation through time via disk-offloaded autograd graph.

Instead of holding all intermediate activations in RAM (OOM on long sequences)
or recomputing them (gradient checkpointing, 2x compute), this module writes
autograd's saved tensors to a binary file during the forward pass and reads
them back during backward. The gradients are EXACT — identical to standard
BPTT with zero approximation.

How it works:
    PyTorch's autograd engine saves intermediate tensors during the forward
    pass so it can compute gradients during backward. We intercept this via
    ``torch.autograd.graph.saved_tensors_hooks``:

    - pack_hook(tensor):   Called when autograd saves a tensor.
                           We write it to disk and return a lightweight handle.
                           The original tensor can be garbage-collected.

    - unpack_hook(handle): Called when backward needs the tensor.
                           We read it from disk and return it.

    Handles survive past the context manager, so backward() can happen later.

    During forward, reads and writes may be interleaved (e.g. per-step
    backward in online RL). This is handled with separate read/write file
    handles and flush-before-read. After forward completes, reads switch
    to a memory-mapped file for efficient random access.

Requires: PyTorch >= 2.0

Example::

    from disk_bptt import DiskBPTT

    disk = DiskBPTT()

    with disk.offload():
        for step in range(1000):
            state = model(state, obs[step])      # saved tensors -> disk
            per_step_loss.backward()              # reads from disk, frees

    full_loss.backward()    # loads remaining tensors from disk
    optimizer.step()
    disk.cleanup()          # delete cache file
"""

__version__ = "0.1.0"

import torch
import os
import mmap
import time
import tempfile
from contextlib import contextmanager


class DiskBPTT:
    """Offload autograd saved tensors to disk for memory-efficient full BPTT.

    Parameters
    ----------
    cache_dir : str or None
        Directory for the cache file. Created if it doesn't exist.
        Defaults to a ``disk_bptt/`` subdirectory in the system temp dir.
    min_bytes : int
        Tensors smaller than this (in bytes) are kept in RAM. Avoids
        disk I/O overhead for tiny tensors like scalars or bias vectors.
        Default ``0`` offloads everything.
    log : bool
        If True, collect timing statistics accessible via :attr:`stats`.
    """

    def __init__(self, cache_dir=None, min_bytes=0, log=False):
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "disk_bptt")
        self.min_bytes = min_bytes
        self.log = log

        self._fpath = None
        self._wf = None   # write handle  (forward: sequential append)
        self._rf = None   # read handle   (random access)
        self._mm = None   # mmap          (post-forward random access)

        self._write_offset = 0
        self._tensor_count = 0
        self._small_kept = 0
        self._fwd_io_ns = 0
        self._bwd_io_ns = 0
        self._bwd_loads = 0
        self._forward_done = False

    # ── Autograd hooks ─────────────────────────────────────────

    def _pack(self, tensor):
        """Autograd hook: write tensor to disk, return handle."""
        nbytes = tensor.nelement() * tensor.element_size()

        if nbytes < self.min_bytes:
            self._small_kept += 1
            return ("ram", tensor.detach().clone())

        t0 = time.monotonic_ns() if self.log else 0

        t = tensor.detach()
        if not t.is_contiguous():
            t = t.contiguous()
        dev = tensor.device
        if dev.type != "cpu":
            t = t.cpu()

        actual = t.nelement() * t.element_size()
        try:
            raw = bytes(t.untyped_storage())[:actual]
        except AttributeError:
            # PyTorch < 2.0 fallback
            raw = t.numpy().tobytes()

        offset = self._write_offset
        self._wf.write(raw)
        self._write_offset += actual
        self._tensor_count += 1

        if self.log:
            self._fwd_io_ns += time.monotonic_ns() - t0

        return ("disk", offset, actual, tuple(t.shape), t.dtype, dev)

    def _unpack(self, handle):
        """Autograd hook: read tensor from disk (or RAM)."""
        if handle[0] == "ram":
            return handle[1]

        _, offset, nbytes, shape, dtype, device = handle

        t0 = time.monotonic_ns() if self.log else 0

        if self._forward_done and self._mm is not None:
            raw = self._mm[offset : offset + nbytes]
        else:
            # Mid-forward: flush write buffer so read handle sees the data
            if self._wf is not None:
                self._wf.flush()
            if self._rf is None:
                self._rf = open(self._fpath, "rb")
            self._rf.seek(offset)
            raw = self._rf.read(nbytes)

        t = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).clone()

        if device.type != "cpu":
            t = t.to(device)

        if self.log:
            self._bwd_io_ns += time.monotonic_ns() - t0
            self._bwd_loads += 1

        return t

    # ── Public API ─────────────────────────────────────────────

    @contextmanager
    def offload(self):
        """Context manager for the forward pass.

        All tensors that autograd saves for backward inside this context
        are written to disk. Handles survive past the context, so
        ``backward()`` can be called after exiting.

        Example::

            with disk.offload():
                loss = model(data)
            loss.backward()
            disk.cleanup()
        """
        self._begin()
        try:
            with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
                yield self
        finally:
            self._finalize_forward()

    def cleanup(self):
        """Delete cache file. Call after ``backward()`` completes."""
        self._close_all()
        if self._fpath and os.path.exists(self._fpath):
            os.remove(self._fpath)
            self._fpath = None

    @property
    def stats(self):
        """Dict of offloading statistics (populated when ``log=True``)."""
        return {
            "tensors_on_disk": self._tensor_count,
            "tensors_in_ram": self._small_kept,
            "disk_mb": self._write_offset / (1024 * 1024),
            "fwd_io_ms": self._fwd_io_ns / 1e6,
            "bwd_io_ms": self._bwd_io_ns / 1e6,
            "bwd_loads": self._bwd_loads,
        }

    # ── Internal ───────────────────────────────────────────────

    def _begin(self):
        self._close_all()
        os.makedirs(self.cache_dir, exist_ok=True)
        self._fpath = os.path.join(self.cache_dir, f"graph_{os.getpid()}.bin")
        if os.path.exists(self._fpath):
            os.remove(self._fpath)
        self._wf = open(self._fpath, "wb", buffering=8 * 1024 * 1024)
        self._write_offset = 0
        self._tensor_count = 0
        self._small_kept = 0
        self._fwd_io_ns = 0
        self._bwd_io_ns = 0
        self._bwd_loads = 0
        self._forward_done = False

    def _finalize_forward(self):
        if self._wf is not None:
            self._wf.flush()
            os.fsync(self._wf.fileno())
            self._wf.close()
            self._wf = None
        if self._rf is not None:
            self._rf.close()
            self._rf = None
        if self._fpath and os.path.exists(self._fpath) and self._write_offset > 0:
            self._rf = open(self._fpath, "rb")
            self._mm = mmap.mmap(self._rf.fileno(), 0, access=mmap.ACCESS_READ)
        self._forward_done = True

    def _close_all(self):
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._rf is not None:
            self._rf.close()
            self._rf = None
        if self._wf is not None:
            self._wf.close()
            self._wf = None

    def __del__(self):
        self._close_all()

    def __repr__(self):
        s = self.stats
        return (
            f"DiskBPTT(tensors={s['tensors_on_disk']}, "
            f"disk={s['disk_mb']:.1f}MB, "
            f"fwd_io={s['fwd_io_ms']:.0f}ms, "
            f"bwd_io={s['bwd_io_ms']:.0f}ms)"
        )
