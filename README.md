<p align="center">
<img width="400" src="/assets/logo.png">
</p>

<h1 align="center">Mini-SGLang + <b>HiCache</b></h1>

<p align="center">
A <b>lightweight yet high-performance</b> LLM inference framework,<br/>
extended with <b>HiCache</b> — a CPU/GPU hierarchical KV cache that uses DRAM as L2 for HBM.
</p>

<p align="center">
  <a href="#-hicache"><b>⭐ HiCache</b></a> ·
  <a href="#-features">Features</a> ·
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-benchmark">Benchmark</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-documentation">Docs</a>
</p>

---

This is a fork of [Mini-SGLang](https://github.com/sgl-project/mini-sglang) (from the SGLang team at LMSYS). The original framework provides a clean, ~5k-line reference implementation of modern LLM serving techniques — Radix Cache, Chunked Prefill, Overlap Scheduling, Tensor Parallelism, CUDA Graph, and FlashAttention/FlashInfer integration.

**This fork adds HiCache**, a full HBM↔DRAM KV cache offloading system, enabling much larger KV cache pools without additional GPU memory.

---

## ⭐ HiCache

**HiCache turns CPU DRAM into a second-level cache for GPU HBM.** KV cache blocks are asynchronously transferred between GPU and CPU, dramatically expanding the effective KV cache capacity while keeping hot data on GPU for fast access.

### The Problem

LLM inference stores per-token Key and Value tensors (the "KV cache") in GPU HBM. As context length and concurrency grow, this cache can consume tens to hundreds of GB. HBM is expensive and limited — but CPU DRAM is abundant and cheap.

### The Solution

HiCache stores KV cache in **two layout options** on CPU, each paired with a GPU-side strategy:

#### Approach 1: CPU `page_first` + GPU `layer_first` (recommended)

CPU pages are stored contiguously for fast write-back; GPU keeps the compute-friendly `layer_first` layout. **Index-based transfer via CUDA kernel** handles the layout mismatch transparently — each token is copied at its correct offset on both sides using independent stride values. No intermediate buffer needed.

```
         CPU (page_first)                     GPU (layer_first)
  ┌────────────────────────┐           ┌────────────────────────┐
  │  Page 0  │ Page 1 │... │           │ L0: [t0][t1][t2]...   │
  │  ┌──────────────┐      │  ──load──▶│ L1: [t0][t1][t2]...   │
  │  │L0 L1 ... LN │      │  ◀─write──│ L2: [t0][t1][t2]...   │
  │  │[tokens 0..S]│      │           │                        │
  │  └──────────────┘      │           │  ← coalesced 访存，计算快  │
  │  ← 整页连续，DMA 友好   │           └────────────────────────┘
  └────────────────────────┘
          ↑ index-based transfer (CUDA kernel, stride-aware)
```

- ✅ GPU 计算不受影响，attention 保持合并访问
- ✅ CPU 整页连续，写回效率高
- ✅ 传输内核自动处理 stride 差异，无需中间媒介

#### Approach 2: CPU `page_first` + GPU `page_first` (pagewise)

Both ends use the same `page_first` layout. Transfer is done via **`torch.Tensor.copy_()` (DMA)**, copying entire pages at once. No CUDA kernel needed. However, attention kernel performance degrades because consecutive tokens span `L × H × D` apart in memory — coalesced access breaks down.

```
         CPU (page_first)                     GPU (page_first)
  ┌────────────────────────┐           ┌────────────────────────┐
  │  Page 0  │ Page 1 │... │  ──DMA──▶│  Page 0  │ Page 1 │... │
  │  ┌──────────────┐      │  ◀──DMA──│  ┌──────────────┐      │
  │  │L0 L1 ... LN │      │           │  │L0 L1 ... LN │      │
  │  └──────────────┘      │           │  └──────────────┘      │
  │  ← 布局一致，整页 copy   │           │  ← 计算访存非连续，慢    │
  └────────────────────────┘           └────────────────────────┘
```

- ✅ 传输零 kernel launch，纯 DMA 拷贝
- ✅ 当 indices 连续时仅需 2 次 `copy_()`（K+V）
- ❌ GPU attention 合并访问失效，计算吞吐下降 20-40%

| 对比维度 | Approach 1 (混合) | Approach 2 (pagewise) |
|----------|:---:|:---:|
| GPU 计算速度 | ⭐⭐⭐ 快 | ⭐ 慢 |
| CPU 写回效率 | ⭐⭐⭐ 快 | ⭐⭐⭐ 快 |
| 传输方式 | CUDA kernel（index + stride） | `copy_()`（DMA） |
| 布局约束 | 各自独立 | 必须一致 |
| **推荐** | ✅ 默认 | 仅极端场景 |

> **核心设计原则**：GPU 计算效率优先于 DMA 带宽，因为计算是持续性的、传输是间歇性且可交叠的。

### Key Mechanisms

| Mechanism | Description |
|-----------|-------------|
| **Three Transport Strategies** | `layerwise` (stream per layer, overlaps compute), `non-layerwise` (bulk DMA for all layers), `pagewise` (page-level copy when layouts match) |
| **Quick Demotion** | After a request finishes, KV cache is asynchronously written to DRAM, then immediately evicted from HBM — freeing GPU memory without blocking compute |
| **Deferred Retry** | If a node is still locked (in use by another request), demotion is deferred and retried at the next write-ack cycle |
| **Race-Free Design** | Demotion gates on `allow_demotion=finished` — never evicts HBM pages still referenced by active decode requests |
| **Index-Based Transfer** | Approach 1 (recommended): CUDA kernel with per-token index + stride, handles `page_first`→`layer_first` layout mismatch without intermediate buffer |
| **Pagewise DMA Transfer** | Approach 2: `torch.Tensor.copy_()` (DMA copy engine), zero kernel launch, only works when both ends share `page_first` layout |

### How to Enable

```bash
# Basic HiCache: host memory = 2× HBM (default)
python -m minisgl --model "Qwen/Qwen3-8B" --cache hiradix

# With Quick Demotion (evict from HBM immediately after DRAM write)
python -m minisgl --model "Qwen/Qwen3-8B" --cache hiradix --hicache-quick-demotion

# Tune host memory ratio
python -m minisgl --model "Qwen/Qwen3-8B" --cache hiradix --hicache-ratio 2.0

# Disable layerwise (use bulk non-layerwise transfer)
python -m minisgl --model "Qwen/Qwen3-8B" --cache hiradix --disable-layerwise
```

---

## ✨ Features

### HiCache (This Fork)
- **HBM↔DRAM KV Cache Offloading** — Use CPU memory as L2 cache for GPU
- **Quick Demotion** — Instant HBM reclamation after async DRAM write
- **Three Transport Modes** — layerwise · non-layerwise · pagewise
- **Deferred Retry** — Graceful handling of locked nodes during demotion
- **Race-Free** — Demotion gated on request completion, never corrupts active handles

### From Upstream (Mini-SGLang)
- **Radix Cache** — Prefix-tree KV reuse across requests
- **Chunked Prefill** — Reduces peak memory for long-context serving
- **Overlap Scheduling** — Hides CPU overhead with GPU computation
- **Tensor Parallelism** — Multi-GPU scaling via NCCL
- **CUDA Graph** — Eliminates CPU launch overhead in decode
- **FlashAttention / FlashInfer** — State-of-the-art attention kernels

---

## 🚀 Quick Start

> ⚠️ **Platform**: Linux only (x86_64 / aarch64). Windows users: use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install).

### 1. Prerequisites

- **Python 3.10+**, **NVIDIA CUDA Toolkit** (match your driver version)
- `uv` (recommended) or `pip`

### 2. Installation

```bash
git clone https://github.com/depictlightning/mini-sglang.git
cd mini-sglang
uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .
```

### 3. Launch with HiCache

```bash
# Basic server with HiCache
python -m minisgl --model "Qwen/Qwen3-0.6B" --cache hiradix

# With Quick Demotion + tuned ratio
python -m minisgl --model "Qwen/Qwen3-8B" --cache hiradix \
    --hicache-ratio 2.0 --hicache-quick-demotion
```

### 4. Interactive Shell

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --cache hiradix --shell
```

<details>
<summary><b>🐳 Docker</b></summary>

```bash
docker build -t minisgl .
docker run --gpus all -p 1919:1919 \
    minisgl --model Qwen/Qwen3-0.6B --cache hiradix --host 0.0.0.0
```
</details>

---

## 📊 Benchmark

### HiCache Throughput

Tested on **1× H800 GPU** with **Qwen3-8B**:

| Configuration | Throughput | vs Baseline |
|---------------|-----------|-------------|
| Radix (HBM only) | baseline | 1.0× |
| **HiRadix (HiCache)** | **1.2×** | **+20%** |

> HiCache achieves 1.2× throughput by offloading cold KV pages to DRAM, freeing HBM for more concurrent requests.

### Upstream Benchmarks

See upstream [bench.py](./benchmark/offline/bench.py) for offline throughput, and [bench_qwen.py](./benchmark/online/bench_qwen.py) for online serving with Qwen3-32B on 4×H200.

---

## 🏗 Architecture

```
User Request
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  API Server  │────▶│  Tokenizer   │────▶│  Scheduler   │
│  (FastAPI)   │◀────│  /Detokenizer│◀────│  (Rank 0)    │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                │ NCCL broadcast
                                     ┌──────────┼──────────┐
                                     ▼          ▼          ▼
                              ┌──────────┐ ┌──────────┐ ┌──────────┐
                              │ Engine 0 │ │ Engine 1 │ │ Engine N │
                              │  (GPU 0) │ │  (GPU 1) │ │  (GPU N) │
                              └────┬─────┘ └──────────┘ └──────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              ┌──────────┐  ┌──────────┐  ┌──────────┐
              │ KV Cache │  │  HiCache │  │ Attention│
              │  (Radix)  │◀─│Controller│─▶│ Backend  │
              └──────────┘  └────┬─────┘  └──────────┘
                                 │ DMA
                                 ▼
                          ┌────────────┐
                          │ CPU DRAM   │
                          │ (Host Pool)│
                          └────────────┘
```

Key modules for HiCache:
- [`python/minisgl/hicache/controller.py`](./python/minisgl/hicache/controller.py) — Transfer orchestration (load/write queues, DMA, Quick Demotion)
- [`python/minisgl/kvcache/hiradix_cache.py`](./python/minisgl/kvcache/hiradix_cache.py) — Dual-tier prefix tree (HBM + DRAM node values)
- [`python/minisgl/scheduler/cache.py`](./python/minisgl/scheduler/cache.py) — CacheManager integration, lazy-free, slot recycling
- [`python/minisgl/kernel/csrc/jit/hicache.cu`](./python/minisgl/kernel/csrc/jit/hicache.cu) — Custom CUDA transfer kernels (legacy; being replaced by DMA)

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [`docs/features.md`](./docs/features.md) | Full feature list & CLI arguments |
| [`docs/structures.md`](./docs/structures.md) | System architecture & process-level data flow |
| [`docs/kv_cache_flow.md`](./docs/kv_cache_flow.md) | 🔍 KV cache full lifecycle: match → allocate → insert → write-back → load → evict |
| [`docs/write_back_chain.md`](./docs/write_back_chain.md) | 🔍 HiCache write-back call chain: from `cache_req` to async DMA ack |

---

## 📄 License

This project inherits the [Apache 2.0 License](./LICENSE) from [Mini-SGLang](https://github.com/sgl-project/mini-sglang).
