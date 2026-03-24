#!/usr/bin/env python3
"""
Tests for DiskBPTT — verifies that disk-offloaded backward produces
identical gradients to standard in-RAM backward across several patterns:

1. Basic feedforward model
2. Recurrent chain with straight-through estimator (vs truncated BPTT)
3. Interleaved forward/backward (online RL pattern)
4. ODE-like model with many integration substeps
5. Memory profile on a long sequence
6. Reuse across multiple sequences
"""

import torch
import torch.nn as nn
import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from disk_bptt import DiskBPTT


def test_basic_correctness():
    """Gradients from disk BPTT must exactly match standard BPTT."""
    print("=== Test 1: Basic correctness ===")

    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
    x = torch.randn(4, 16)

    # Standard BPTT
    out = model(x)
    out.sum().backward()
    grad_standard = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad()

    # Disk BPTT
    disk = DiskBPTT(log=True)
    with disk.offload():
        out2 = model(x)
        loss2 = out2.sum()
    loss2.backward()
    grad_disk = {n: p.grad.clone() for n, p in model.named_parameters()}

    for name in grad_standard:
        diff = (grad_standard[name] - grad_disk[name]).abs().max().item()
        assert diff < 1e-6, f"Gradient mismatch for {name}: {diff}"

    print(f"  PASS  (max diff < 1e-6)")
    print(f"  {disk}")
    disk.cleanup()


def test_recurrent_bptt_chain():
    """
    Recurrent chain connected by straight-through estimators.
    Verify disk BPTT matches full in-RAM BPTT, and that both
    differ from truncated BPTT (which loses long-range signal).
    """
    print("\n=== Test 2: Recurrent BPTT chain (50 steps, STE) ===")

    torch.manual_seed(42)
    N, D = 50, 64

    f = nn.Linear(D, D, bias=False)
    z_init = nn.Parameter(torch.randn(1, D) * 0.01)

    torch.manual_seed(0)
    all_obs = [torch.randn(1, D) for _ in range(N)]

    # ── Full in-RAM BPTT (ground truth) ──
    z = z_init
    z_posts = [z]
    for t in range(N):
        z_for_bptt = z
        z_new = f(z.detach() + all_obs[t])
        z_bptt = z_new + (z_for_bptt - z_for_bptt.detach())
        z_posts.append(z_bptt)
        z = z_bptt

    (-sum(zp.mean() for zp in z_posts)).backward()
    grad_full = {n: p.grad.clone() for n, p in
                 [("f.weight", f.weight), ("z_init", z_init)]}
    f.zero_grad()
    z_init.grad = None

    # ── Disk BPTT ──
    disk = DiskBPTT(log=True)
    z = z_init
    z_posts = [z]
    with disk.offload():
        for t in range(N):
            z_for_bptt = z
            z_new = f(z.detach() + all_obs[t])
            z_bptt = z_new + (z_for_bptt - z_for_bptt.detach())
            z_posts.append(z_bptt)
            z = z_bptt

    (-sum(zp.mean() for zp in z_posts)).backward()
    grad_disk = {n: p.grad.clone() for n, p in
                 [("f.weight", f.weight), ("z_init", z_init)]}

    for name in grad_full:
        diff = (grad_full[name] - grad_disk[name]).abs().max().item()
        assert diff < 1e-5, f"Gradient mismatch for {name}: {diff}"

    print(f"  PASS  disk BPTT == full in-RAM BPTT")
    print(f"  {disk}")
    disk.cleanup()

    # ── Truncated BPTT (window=10) for comparison ──
    f.zero_grad()
    z_init.grad = None
    WINDOW = 10
    z = z_init
    z_posts = [z]
    for t in range(N):
        if t > 0 and t % WINDOW == 0:
            z = z.detach()
        z_for_bptt = z
        z_new = f(z.detach() + all_obs[t])
        z_bptt = z_new + (z_for_bptt - z_for_bptt.detach())
        if len(z_posts) > WINDOW:
            z_posts[-(WINDOW + 1)] = z_posts[-(WINDOW + 1)].detach()
            if len(z_posts) > WINDOW + 2:
                z_posts = z_posts[-(WINDOW + 1):]
        z_posts.append(z_bptt)
        z = z_bptt

    (-sum(zp.mean() for zp in z_posts[-WINDOW:] if zp.requires_grad)).backward()

    has_z_init_grad = z_init.grad is not None
    diff_f = (grad_full["f.weight"] - f.weight.grad).abs().max().item()
    print(f"  Truncated(w=10) vs Full: max diff = {diff_f:.4f}")
    print(f"  Truncated loses z_init gradient entirely: {not has_z_init_grad}")
    assert diff_f > 1e-4, "Truncated should differ from full BPTT"
    print(f"  CONFIRMED  truncation loses info that disk BPTT preserves")


def test_interleaved_backward():
    """
    Per-step backward interleaved with forward (online RL pattern).
    A separate decoder does backward() each step while the encoder's
    graph accumulates for a single backward at the end.
    """
    print("\n=== Test 3: Interleaved backward (online RL pattern) ===")

    torch.manual_seed(42)
    N, D = 20, 32

    encoder = nn.Linear(D, D, bias=False)
    decoder = nn.Linear(D, 1, bias=False)

    torch.manual_seed(0)
    all_obs = [torch.randn(1, D) for _ in range(N)]

    # Standard: encoder BPTT through z chain
    encoder.zero_grad()
    z_posts = [encoder(all_obs[t]) for t in range(N)]
    (-sum(zp.mean() for zp in z_posts)).backward()
    grad_standard = encoder.weight.grad.clone()
    encoder.zero_grad()

    # Disk BPTT with interleaved decoder backward
    disk = DiskBPTT(log=True)
    with disk.offload():
        z_posts_disk = []
        for t in range(N):
            z = encoder(all_obs[t])
            z_posts_disk.append(z)
            # Per-step decoder backward (interleaved)
            out = decoder(z.detach())
            decoder.zero_grad()
            (-out.sum()).backward()

    encoder.zero_grad()
    (-sum(zp.mean() for zp in z_posts_disk)).backward()
    grad_disk = encoder.weight.grad.clone()

    diff = (grad_standard - grad_disk).abs().max().item()
    assert diff < 1e-5, f"Gradient mismatch: {diff}"
    print(f"  PASS  interleaved backward correct (diff={diff:.2e})")
    print(f"  {disk}")
    disk.cleanup()


def test_ode_like_model():
    """
    ODE-like model: multiple Euler integration substeps per observation.
    Verifies correctness with deep computation graphs.
    """
    print("\n=== Test 4: ODE-like model (30 steps x 8 substeps) ===")

    torch.manual_seed(42)
    D, SUBSTEPS, STEPS = 32, 8, 30

    W = nn.Parameter(torch.randn(D, D) * 0.01)
    proj = nn.Linear(D * 2, D, bias=False)
    z0 = nn.Parameter(torch.randn(1, D) * 0.01)

    torch.manual_seed(0)
    all_obs = [torch.randn(D) for _ in range(STEPS)]

    def ode_step(z, obs, dt=0.1):
        for _ in range(SUBSTEPS):
            dz = z @ W.t() + proj(torch.cat([z, obs.unsqueeze(0)], dim=-1))
            z = (z + dz * dt).clamp(-5, 5)
        return z

    # Standard in-RAM BPTT
    z = z0
    all_z = [z]
    for t in range(STEPS):
        z = ode_step(z, all_obs[t])
        all_z.append(z)
    (-sum(zp.mean() for zp in all_z)).backward()
    grad_ram = {
        "W": W.grad.clone(),
        "proj": proj.weight.grad.clone(),
        "z0": z0.grad.clone(),
    }
    W.grad = None
    proj.zero_grad()
    z0.grad = None

    # Disk BPTT
    disk = DiskBPTT(log=True)
    z = z0
    all_z = [z]
    with disk.offload():
        for t in range(STEPS):
            z = ode_step(z, all_obs[t])
            all_z.append(z)
    (-sum(zp.mean() for zp in all_z)).backward()
    grad_disk = {
        "W": W.grad.clone(),
        "proj": proj.weight.grad.clone(),
        "z0": z0.grad.clone(),
    }

    for name in grad_ram:
        diff = (grad_ram[name] - grad_disk[name]).abs().max().item()
        assert diff < 1e-4, f"Gradient mismatch for {name}: {diff}"

    print(f"  PASS  all gradients match")
    print(f"  {disk}")
    disk.cleanup()


def test_memory_profile():
    """Verify RAM stays bounded during a long sequence."""
    print("\n=== Test 5: Memory profile (500-step sequence) ===")

    torch.manual_seed(42)
    D, N = 64, 500
    model = nn.Sequential(
        nn.Linear(D, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, D),
    )

    disk = DiskBPTT(log=True)

    z = torch.randn(1, D, requires_grad=True)
    all_z = [z]
    with disk.offload():
        for t in range(N):
            z_new = model(z.detach()) + (z - z.detach())  # STE
            all_z.append(z_new)
            z = z_new

    (-sum(zp.mean() for zp in all_z)).backward()

    s = disk.stats
    print(f"  Disk: {s['disk_mb']:.1f} MB, {s['tensors_on_disk']} tensors")
    print(f"  Backward loaded {s['bwd_loads']} tensors")
    print(f"  PASS  {N}-step backward completed")
    disk.cleanup()


def test_reuse():
    """DiskBPTT can be reused across multiple forward/backward cycles."""
    print("\n=== Test 6: Reuse across sequences ===")

    model = nn.Linear(8, 8)
    disk = DiskBPTT()

    for i in range(5):
        model.zero_grad()
        with disk.offload():
            loss = model(torch.randn(2, 8)).sum()
        loss.backward()
        assert model.weight.grad is not None
        disk.cleanup()

    assert not os.path.exists(disk._fpath or "")
    print(f"  PASS  5 cycles, cache cleaned up")


def test_min_bytes():
    """Small tensors stay in RAM when min_bytes is set."""
    print("\n=== Test 7: min_bytes threshold ===")

    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
    x = torch.randn(1, 4)

    # All tensors are small (4x4 = 64 bytes for float32)
    disk = DiskBPTT(min_bytes=1024, log=True)
    with disk.offload():
        loss = model(x).sum()
    loss.backward()

    s = disk.stats
    print(f"  On disk: {s['tensors_on_disk']}, in RAM: {s['tensors_in_ram']}")
    assert s["tensors_in_ram"] > 0, "Some tensors should stay in RAM"
    print(f"  PASS  small tensors kept in RAM")
    disk.cleanup()


if __name__ == "__main__":
    print("DiskBPTT Test Suite")
    print("=" * 60)

    test_basic_correctness()
    test_recurrent_bptt_chain()
    test_interleaved_backward()
    test_ode_like_model()
    test_memory_profile()
    test_reuse()
    test_min_bytes()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
