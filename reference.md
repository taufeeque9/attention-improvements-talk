# Reference Calculations

Detailed derivations of the numbers used in `slides/slides.md`. Verify here before the talk.

All FP arithmetic counts: a multiply-accumulate (MAC) = 2 FLOPs.
All matmul $A \in \mathbb{R}^{m \times k}, B \in \mathbb{R}^{k \times n}$: $A B$ costs $2 m n k$ FLOPs.

---

## GPT-2 family architectures (Radford et al. 2019)

| Variant | params | $L$ | $H$ | $d$ | $d_k = d/H$ | native $n$ |
|---|---|---|---|---|---|---|
| small | 124 M | 12 | 12 | 768 | 64 | 1024 |
| medium | 355 M | 24 | 16 | 1024 | 64 | 1024 |
| **large** | **774 M** | **36** | **20** | **1280** | **64** | **1024** |
| XL | 1.5 B | 48 | 25 | 1600 | 64 | 1024 |

**Slides use GPT-2 large.** Verified against HuggingFace `openai-community/gpt2-large/config.json`: `n_layer=36`, `n_head=20`, `n_embd=1280`, `n_ctx=1024`. Params 774M per HF model card (matches Radford et al. 2019 Table 2).

---

## Attention prefill FLOPs (GPT-2 large)

**Per layer**, attention has two matmuls in the softmax core:

1. $Q K^\top$: per head, $(n, d_k) \times (d_k, n) \to (n, n)$. FLOPs $= 2 n^2 d_k$. Over $H$ heads: $2 H n^2 d_k$.
2. $(\text{softmax}\,QK^\top)\, V$: per head, $(n, n) \times (n, d_k) \to (n, d_k)$. FLOPs $= 2 n^2 d_k$. Over $H$ heads: $2 H n^2 d_k$.

Total per layer (ignoring softmax + projections): $4 H n^2 d_k$.

For GPT-2 large: $4 \times 20 \times 64 = 5120$, so per layer $= 5120\, n^2$ FLOPs.

Over $L = 36$ layers: $\;\;184{,}320\, n^2 \;\;\approx\; 1.84 \times 10^5 \, n^2$ FLOPs.

### Plugged in

| $n$ | $1.84 \times 10^5 \cdot n^2$ | Rounded |
|---|---|---|
| $1024 \;(\approx 10^3)$ | $1.93 \times 10^{11}$ | **193 GFLOPs** |
| $10^5$ | $1.84 \times 10^{15}$ | **1.8 PFLOPs** |
| $10^6$ | $1.84 \times 10^{17}$ | **184 PFLOPs** |

(Caveat: this counts only the core softmax attention. Adding the four projections $W_Q, W_K, W_V, W_O$ each is $2 n d^2 = 2 n \cdot 1280^2$ FLOPs — comparatively small at long context, dominant at short context. Slide focuses on the $O(n^2)$ piece, which is the part that breaks.)

---

## KV cache size (GPT-2 large, FP16)

Per token, per layer: $2 \,(\text{K and V}) \times H \times d_k \times 2 \text{ bytes (fp16)}$
$\;\;= 2 \times 20 \times 64 \times 2 \;=\; 5120$ B = 5 KB / token / layer.

Over $L = 36$ layers: $\;5120 \times 36 \;=\; 184{,}320$ B $\;\approx\; 180$ KB **per token**.

| $n$ | KV total |
|---|---|
| $1024$ | 184 MB |
| $10^5$ | 18 GB |
| $10^6$ | **184 GB** |

---

## Hardware reference: NVIDIA H100 SXM5

Verified against [NVIDIA H100 product page](https://www.nvidia.com/en-us/data-center/h100/). NVIDIA's marketing numbers are "with sparsity" (2× dense); dense figures here halve those.

| Spec | Value |
|---|---|
| FP16 dense tensor TFLOPs | 989 TFLOPs ≈ **1 PFLOPs/s** (1979 with sparsity) |
| FP8 dense tensor TFLOPs | ~1979 TFLOPs (3958 with sparsity) |
| HBM3 bandwidth | **3.35 TB/s** |
| HBM3 capacity | **80 GB** |

### Derived numbers used on the slide

- **Prefill attention at $n=10^6$**: $1.84 \times 10^{17}$ FLOPs / $10^{15}$ FLOPs/s $\;\approx\;$ **184 s** (~3 min) on one GPU, attention math only.
- **KV cache at $n=10^6$**: 184 GB > 80 GB → **doesn't fit on one H100**. Need $\lceil 184/80 \rceil = 3$ GPUs just to hold the cache.
- **Decode bandwidth wall at $n=10^6$**: 184 GB streamed per generated token at 3.35 TB/s $\;\approx\;$ 55 ms/token, $\;\approx\;$ **18 tokens / s** ceiling.

---

## Comparison: Llama 3 70B (full-MHA hypothetical)

Llama 3 70B uses GQA (8 KV heads). For both the "imagine if we used MHA" and "actual production" comparisons:

| Symbol | Value |
|---|---|
| $L$ | 80 |
| $H$ | 64 |
| $d_k$ | 128 |
| $d$ | 8192 |

### KV cache per token at $n=10^6$ (fp16)

| Scheme | per layer per token | total/token | at $n=10^6$ |
|---|---|---|---|
| **Full MHA** | $2 \times 64 \times 128 \times 2 = 32$ KB | $80 \times 32 = $ 2.56 MB | **~2.6 TB** |
| **GQA (8 KV heads, as published)** | $2 \times 8 \times 128 \times 2 = 4$ KB | $80 \times 4 = $ 320 KB | **~330 GB** |

The full-MHA KV cache for Llama 3 70B at 1M context would be 2.6 TB — more than 30 H100s. GQA brings it to 330 GB (still 4+ H100s, but a different order of magnitude).

### Attention FLOPs/token ratio vs GPT-2 large

The per-token FLOPs factor (attention only) is $L \cdot H \cdot d_k$:

- GPT-2 large: $36 \times 20 \times 64 = 46{,}080$
- Llama 3 70B: $80 \times 64 \times 128 = 655{,}360$
- **Ratio: ~14×**

---

## Comparison: Llama 3 8B (for completeness)

Llama 3 8B uses GQA (8 KV heads). Same calc:

| Symbol | Value |
|---|---|
| $L$ | 32 |
| $H$ | 32 |
| $d_k$ | 128 |
| $d$ | 4096 |

| Scheme | per layer per token | total/token | at $n=10^6$ |
|---|---|---|---|
| Full MHA | $2 \times 32 \times 128 \times 2 = 16$ KB | $32 \times 16 = $ 512 KB | **~524 GB** |
| GQA (8 KV) | $2 \times 8 \times 128 \times 2 = 4$ KB | $32 \times 4 = $ 128 KB | **~128 GB** |

Attention FLOPs/token factor: $32 \times 32 \times 128 = 131{,}072$ → 2.8× more than GPT-2 large.

---

## What this section is for

Every number on the "Concrete: GPT-2 at the agentic edge" slide and its amber callout traces to a derivation above. Spot-check before presenting:

- [x] GPT-2 large architecture verified from HF `openai-community/gpt2-large/config.json` (L=36, H=20, d=1280, n_ctx=1024).
- [x] Llama 3 70B architecture verified from Llama 3 herd paper Table 3 / Section 3.2 ([arXiv:2407.21783](https://arxiv.org/abs/2407.21783)): L=80, H=64, d=8192, num_kv_heads=8.
- [x] H100 SXM5 specs verified from [NVIDIA H100 product page](https://www.nvidia.com/en-us/data-center/h100/): 989 TFLOPS FP16 dense, 80 GB, 3.35 TB/s.
- [ ] Sanity-check the 184 s prefill claim: ignoring projection FLOPs ($O(nd^2)$, dominated by attention at long context); ignoring kernel launch / softmax / mask overhead; assumes 100% HBM bandwidth utilization and 100% tensor-core utilization (both optimistic).
- [ ] Sanity-check the 18 tok/s decode ceiling: assumes (a) the FULL KV cache is streamed per token (true for naive MHA decode), (b) we're at peak HBM bandwidth (optimistic), (c) no batching across requests.

---

## Sources / formulas

- The 2-FLOPs-per-MAC convention is standard; cited in Kaplan 2020 ("Scaling Laws for Neural Language Models") and elsewhere.
- The attention FLOPs decomposition matches Chinchilla / FLOP-counting conventions (Hoffmann et al. 2022).
- The KV cache size formula is standard and matches HuggingFace `model.config` × `seq_len` × 2 (K+V) × 2 bytes (fp16) calculations.
