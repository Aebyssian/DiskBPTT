# DiskBPTT

Full-sequence backpropagation through time with disk-offloaded autograd.

```
              Standard BPTT    Grad Checkpoint    DiskBPTT
 RAM          O(T * model)     O(sqrt(T) * model) O(model)
 Disk         0                0                  O(T * model)
 Compute      1x fwd + 1x bwd 2x fwd + 1x bwd   1x fwd + 1x bwd
 Gradients    exact            exact              exact
```

DiskBPTT gives you the gradients of standard BPTT at the memory cost of truncated BPTT. No recomputation or approximation necessary. It intercepts PyTorch's autograd engine and saves intermediate tensors to disk instead of holding them in RAM, then memory-maps them back during backward.

## Install

```bash
pip install disk-bptt
```

(Or just copy `disk_bptt.py`)

## Usage

```python
from disk_bptt import DiskBPTT

disk = DiskBPTT()

with disk.offload():
    for step in range(1000):
        state = model(state, obs[step])

loss.backward()     # tensors loaded from disk as needed
optimizer.step()
disk.cleanup()      # delete cache file
```

## When to use this

You're doing BPTT through a long sequence and running out of RAM. Your options:

1. **Truncated BPTT**: detach every N steps. Loses long-range gradient signal. 

2. **Gradient checkpointing** (`torch.utils.checkpoint`) — recompute forward pass segments during backward. Saves RAM but costs 2x compute. Doesn't work as well with stateful models.

3. **DiskBPTT**: write saved tensors to disk during forward, read them back during backward. Same RAM as truncated BPTT, same gradients as full BPTT, same compute as standard BPTT. Costs disk I/O (fast on SSDs).

DiskBPTT is the right choice when:

-You need exact full-sequence gradients

-You can't afford the 2x forward compute of gradient checkpointing

-You have an SSD with space for the activations

## How it works

PyTorch's autograd saves tensors during forward so it can compute gradients during backward. DiskBPTT hooks into this mechanism via [`saved_tensors_hooks`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.graph.saved_tensors_hooks):

```
Forward pass:
  autograd wants to save tensor → pack_hook fires → write to disk → return handle
  (tensor freed from RAM)

Backward pass:
  autograd needs saved tensor → unpack_hook fires → read from disk → return tensor
  (tensor freed after op completes)
```

At any point during backward, only one operation's saved tensors are in RAM. Peak memory is bounded by the largest single operation, not the sequence length.

### Interleaved read/write

Some training patterns (e.g. online RL with per-step policy updates) call `backward()` during the forward pass. DiskBPTT handles this with separate read/write file handles and flush-before-read, then switches to memory-mapped I/O after forward completes

## API

### `DiskBPTT(cache_dir=None, min_bytes=0, log=False)`

**cache_dir**: where to put the cache file. Defaults to `/tmp/disk_bptt/`.
**min_bytes**: tensors smaller than this stay in RAM. Useful to avoid I/O overhead for scalars and bias vectors. Default 0 offloads everything.
**log**: collect timing stats in `.stats`.

### `.offload()` → context manager

Wrap your forward pass. All autograd saved tensors inside the context go to disk.

### `.cleanup()`

Delete the cache file. Call after `backward()` completes.

### `.stats` → dict

When `log=True`:
```python
{
    "tensors_on_disk": 1200,
    "tensors_in_ram": 5,
    "disk_mb": 48.0,
    "fwd_io_ms": 120.5,
    "bwd_io_ms": 45.2,
    "bwd_loads": 1200,
}
```

## Examples

### RNN / Recurrent model

```python
disk = DiskBPTT()
hidden = model.init_hidden()

with disk.offload():
    for t in range(seq_len):
        output, hidden = model(input[t], hidden)
        outputs.append(output)

loss = criterion(torch.stack(outputs), target)
loss.backward()      # full BPTT through all seq_len steps
optimizer.step()
disk.cleanup()
```

### Neural ODE

```python
disk = DiskBPTT()

with disk.offload():
    # ODE solver saves many intermediate states
    trajectory = odeint(func, y0, t, method='dopri5')

loss = trajectory[-1].sum()
loss.backward()      # gradients through full trajectory
disk.cleanup()
```

### Online RL with per-step updates

```python
disk = DiskBPTT()
encoder_states = []

with disk.offload():
    for step in range(episode_length):
        z = encoder(obs)
        encoder_states.append(z)

        # Decoder backward happens each step (interleaved)
        action = decoder(z.detach())
        decoder_loss = -advantage * action.log_prob
        decoder_optimizer.zero_grad()
        decoder_loss.backward()
        decoder_optimizer.step()

# Encoder backward through full episode
encoder_loss = compute_encoder_loss(encoder_states)
encoder_optimizer.zero_grad()
encoder_loss.backward()     # loads encoder tensors from disk
encoder_optimizer.step()
disk.cleanup()
```

## Limitations

-Cache file is not cleaned up if the process crashes. check `/tmp/disk_bptt/`

-No deduplication. if autograd saves the same tensor twice, it's written twice

## Test Script

```bash
python test_disk_bptt.py
```

Runs 7 tests covering basic correctness, recurrent chains, interleaved backward, ODE models, memory profiles, reuse, and the min_bytes threshold
