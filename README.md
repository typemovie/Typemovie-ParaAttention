# Typemovie-ParaAttention

TypeMovie-ParaAttention is an enhanced version of ParaAttention, designed to accelerate Diffusion Transformer (DiT) model inference with context parallelism, dynamic caching, and a new high-performance SageAttention backend. It supports both Ulysses Style and Ring Style parallelism, delivering faster inference without sacrificing accuracy.

## What's New

- **SageAttention Backend**: Replaced the original attention implementation with SageAttention, achieving up to **50% faster performance** compared to FlashAttention2 on RTX 4090, while maintaining full compatibility with ParaAttention's API.
- **Enhanced UnifiedAttnMode**: Optimized context parallelism with a hybrid of Ulysses and Ring styles for maximum flexibility and performance across various models and hardware configurations.
- **Seamless Compatibility**: Use all existing ParaAttention interfaces without code changes, ensuring a smooth transition to TypeMovie-ParaAttentio

🔥[Fastest FLUX.1-dev Inference with Context Parallelism and First Block Cache on NVIDIA L20 GPUs](./doc/fastest_flux.md)🔥

🔥[Fastest HunyuanVideo Inference with Context Parallelism and First Block Cache on NVIDIA L20 GPUs](./doc/fastest_hunyuan_video.md)🔥

# Key Features

### Context Parallelism

**Context Parallelism (CP)** partitions neural network activations across multiple GPUs along the sequence dimension, parallelizing all layers for optimal performance. TypeMovie-ParaAttention introduces **UnifiedAttnMode**, combining Ulysses and Ring parallelism for superior efficiency.

Enable context parallelism with a single function call for diffusers pipelines:

```python
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe

parallelize_pipe(pipe)
```

### First Block Cache (Our Dynamic Caching)

Inspired by TeaCache](https://github.com/ali-vilab/TeaCache), First Block Cache (FBCache) uses the residual output of the first transformer block as a cache indicator. If the difference between consecutive residual outputs is small, subsequent transformer block computations are skipped, achieving up to 2x speedup with minimal accuracy loss.

#### Optimizations for FLUX Image Generation Model on a Single NVIDIA L20 GPU

| Optimizations | Original | FBCache rdt=0.06 | FBCache rdt=0.08 | FBCache rdt=0.10 | FBCache rdt=0.12 |
| - | - | - | - | - | - |
| Preview | ![Original](./assets/flux_original.png) | ![FBCache rdt=0.06](./assets/flux_fbc_0.06.png) | ![FBCache rdt=0.08](./assets/flux_fbc_0.08.png) | ![FBCache rdt=0.10](./assets/flux_fbc_0.10.png) | ![FBCache rdt=0.12](./assets/flux_fbc_0.12.png) |
| Wall Time (s) | 26.36 | 21.83 | 17.01 | 16.00 | 13.78 |

#### Optimizations for Video Models

| Model | Optimizations | Preview |
| - | - | - |
| HunyuanVideo | Original | [Original](https://github.com/user-attachments/assets/883d771a-e74e-4081-aa2a-416985d6c713) |
| HunyuanVideo | FBCache | [FBCache](https://github.com/user-attachments/assets/f77c2f58-2b59-4dd1-a06a-a36974cb1e40) |

You only need to call a single function to enable First Block Cache on your `diffusers` pipeline:

```python
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(
    pipe,
    # residual_diff_threshold=0.0,
)
```

Adjust the `residual_diff_threshold` to balance the speedup and the accuracy.
Higher `residual_diff_threshold` will lead to more cache hits and higher speedup, but might also lead to a higher accuracy drop.

# Officially Supported Models

## Context Parallelism with First Block Cache

You could run the following examples with `torchrun` to enable context parallelism with dynamic caching.
You can modify the code to enable `torch.compile` to further accelerate the model inference.
If you want quantization, please refer to [diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao) for more information.
For example, to run FLUX with 2 GPUs:

**Note**: To measure the performance correctly with `torch.compile`, you need to warm up the model by running it for a few iterations before measuring the performance.

```bash
# Use --nproc_per_node to specify the number of GPUs
torchrun --nproc_per_node=2 parallel_examples/run_flux.py
```

- [FLUX🚀](parallel_examples/run_flux.py)
- [HunyuanVideo🚀](parallel_examples/run_hunyuan_video.py)
- [Mochi](parallel_examples/run_mochi.py)
- [CogVideoX](parallel_examples/run_cogvideox.py)
- [Wan](parallel_examples/run_wan.py)   

## Single GPU Inference with First Block Cache

You can also run the following examples with a single GPU and enable the First Block Cache to speed up the model inference.

```bash
python3 first_block_cache_examples/run_hunyuan_video.py
```

- [FLUX🚀](first_block_cache_examples/run_flux.py)
- [HunyuanVideo🚀](first_block_cache_examples/run_hunyuan_video.py)
- [Mochi](first_block_cache_examples/run_mochi.py)
- [CogVideoX](first_block_cache_examples/run_cogvideox.py)
- [Wan](first_block_cache_examples/run_wan.py)

# Installation

For NVIDIA RTX 4090 or 5090 GPUs, install the SageAttention backend for optimal performance:

```bash
git clone https://github.com/umerkay/SageAttention.git
cd SageAttention
python3 setup.py install
```

## Install from PyPI

```bash
pip3 install typemovie-paraattention
```

## Local Installation

```bash
git clone https://github.com/typemovie/Typemovie-ParaAttention
cd ParaAttention
#git submodule update --init --recursive

python3 setup.py install
```

# Usage

## All Examples

Please refer to examples in the `parallel_examples` and `first_block_cache_examples` directories.

### Parallelize Models

| Model | Command |
| - | - |
| `FLUX` | `torchrun --nproc_per_node=2 parallel_examples/run_flux.py` |
| `HunyuanVideo` | `torchrun --nproc_per_node=2 parallel_examples/run_hunyuan_video.py` |
| `Mochi` | `torchrun --nproc_per_node=2 parallel_examples/run_mochi.py` |
| `CogVideoX` | `torchrun --nproc_per_node=2 parallel_examples/run_cogvideox.py` |

### Apply First Block Cache

| Model | Command |
| - | - |
| `FLUX` | `python3 first_block_cache_examples/run_flux.py` |
| `HunyuanVideo` | `python3 first_block_cache_examples/run_hunyuan_video.py` |
| `Mochi` | `python3 first_block_cache_examples/run_mochi.py` |
| `CogVideoX` | `python3 first_block_cache_examples/run_cogvideox.py` |

## Parallelize VAE

VAE can be parallelized with `para_attn.parallel_vae.diffusers_adapters.parallelize_vae`.
Currently, only `AutoencoderKL` and `AutoencoderKLHunyuanVideo` are supported.

``` python
import torch
import torch.distributed as dist
from diffusers import AutoencoderKL

dist.init_process_group()

torch.cuda.set_device(dist.get_rank())

vae = AutoencoderKL.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

parallelize_vae(vae)
```

## Run Unified Attention (Hybird Ulysses Style and Ring Style) with `torch.compile`

```python
import torch
import torch.distributed as dist
import torch.nn.functional as F
from para_attn import para_attn_interface

dist.init_process_group()
world_size = dist.get_world_size()
rank = dist.get_rank()

assert world_size <= torch.cuda.device_count()
if world_size % 2 == 0:
    mesh_shape = (2, world_size // 2)
else:
    mesh_shape = (1, world_size)

B, H, S_Q, S_KV, D = 2, 24, 4096, 4096, 64
dtype = torch.float16
device = "cuda"

def func(*args, **kwargs):
    return F.scaled_dot_product_attention(*args, **kwargs)

# torch._inductor.config.reorder_for_compute_comm_overlap = True
# func = torch.compile(func)

with torch.no_grad(), torch.cuda.device(rank):
    torch.manual_seed(0)

    query = torch.randn(B, H, S_Q, D, dtype=dtype, device=device)
    key = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)
    value = torch.randn(B, H, S_KV, D, dtype=dtype, device=device)
    attn_mask = None
    dropout_p = 0.0
    is_causal = False

    query_slice = query.chunk(world_size, dim=-2)[rank]
    key_slice = key.chunk(world_size, dim=-2)[rank]
    value_slice = value.chunk(world_size, dim=-2)[rank]

    for _ in range(2):
        mesh = dist.init_device_mesh(device, mesh_shape, mesh_dim_names=("ring", "ulysses"))
        with para_attn_interface.UnifiedAttnMode(mesh):
            out_slice = func(
                query_slice,
                key_slice,
                value_slice,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

    out_slice_ref = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    ).chunk(world_size, dim=-2)[rank]

    torch.testing.assert_close(out_slice, out_slice_ref, rtol=1e-5, atol=1e-3 * world_size)

dist.destroy_process_group()
```

Save the above code to `test.py` and run it with `torchrun`:

```bash
torchrun --nproc_per_node=2 test.py
```

