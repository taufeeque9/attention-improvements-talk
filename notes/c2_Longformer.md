# c2 — Longformer

**Paper**: Beltagy, Peters, Cohan (Allen Institute for AI), *Longformer: The Long-Document Transformer*, EMNLP 2020. [arXiv:2004.05150](https://arxiv.org/abs/2004.05150). [PDF](../papers/c2_Longformer_SWA.pdf).
**Length**: 17 pages. Read §1-§4.2 + Figures 1-2 + Tables 1-3 (pp. 1-5) this session.
**Why it matters**: **The canonical "sliding window attention" (SWA) paper.** Mistral 7B, Gemma 2/3, Llama 4, Command A all use direct descendants of Longformer's SWA pattern. Replaces Sparse Transformer (c1) as our (c)-section anchor.

---

## 1. Bottleneck targeted

**Same as Sparse Transformer**: $O(n^2)$ attention compute. Authors want to process documents of 4K–32K tokens — modest by 2026 standards but a real wall in 2020 when BERT was capped at 512 and Transformer-XL was the long-context option.

Long-document NLP tasks (QA over Wikipedia articles, multi-document summarization, coreference over entire documents) need >512-token context but had to chunk-and-aggregate, losing cross-chunk information. Longformer is the "drop-in replacement for self-attention" that lets you skip the chunking.

## 2. Mechanism — three attention patterns

Figure 2 is the canonical visual: four panels showing full attention vs the three Longformer variants.

### 2a. Sliding window attention (SWA) — Fig 2b

Each token attends to **$w/2$ tokens on each side** (total window $w$). Cost: **$O(n \cdot w)$**.

Receptive field stacked across $\ell$ layers: $\ell \cdot w$ tokens. So a 24-layer model with $w = 512$ can see ~12K tokens of effective context, even though each layer is only $O(n \cdot 512)$.

**Window size varies per layer** (§4.1): small in low layers (local context), large in high layers (cross-document reasoning). This is the design choice that survived into Gemma's local-global pattern.

### 2b. Dilated sliding window — Fig 2c

Same as SWA but the window has **gaps of size $d$** (like dilated CNNs). Receptive field at $\ell$ layers: $\ell \cdot d \cdot w$. Used only on a few heads in higher layers; most heads remain non-dilated.

This is conceptually close to Sparse Transformer's strided pattern, but applied as a multi-head specialization rather than a per-layer choice.

### 2c. Global + sliding window — Fig 2d

A small set of **task-specific tokens get global attention** (attend to all positions; all positions attend to them). Examples:
- `[CLS]` for classification
- Question tokens for QA
- Doc/sentence boundary tokens

**Symmetric**: global tokens both read from and write to everything. Complexity stays $O(n)$ because the number of global tokens $|G|$ is constant in $n$.

**Critical implementation detail (§3.1)**: global attention uses **separate Q, K, V projections** ($Q_g, K_g, V_g$) from the local SWA attention ($Q_s, K_s, V_s$). The $Q_g$ etc. are initialized to match $Q_s$ etc. (no cold start). This separation lets the model learn distinct local vs global semantics.

## 3. Cost reduction — claimed and measured

**Asymptotic**: $O(n^2) \to O(n \cdot w)$. For $w = 512, n = 16384$: 32× less attention compute than dense.

**Wall-clock** (Figure 1, custom CUDA kernel):

| Implementation | Notes |
|---|---|
| `Longformer-loop` | Pure PyTorch, very slow — testing only |
| `Longformer-chunks` | Vectorized non-dilated — used for pretraining/finetuning |
| `Longformer-cuda` | Custom TVM-compiled CUDA kernel for dilated case |

The headline plot: full self-attention runs OOM by $n \approx 4000$ on a V100; Longformer-cuda scales linearly up to $n = 16384$ with constant memory growth. Time-per-batch grows linearly vs quadratically.

**Quality** (Tables 2-3, character-level LM):

| Model | text8 BPC | enwik8 BPC | Params |
|---|---|---|---|
| Transformer-XL (24L) | — | 0.99 | 277M |
| **Sparse Transformer (Child 2019)** | — | **0.99** | ~100M |
| Adaptive (Sukhbaatar 2019) | 1.05 | 1.02 | 209M |
| Compressive (Rae 2020) | — | 0.97 | 277M |
| **Longformer (large)** | 1.10 / 1.04 | **0.99** | 102M / 41M |

Longformer matches Sparse Transformer at 102M params and matches Compressive (0.97) within 0.02 BPC at 1/3 the parameter count.

## 4. Hardware-aware aspects

§3.2 implementation:
- The expensive op is the $QK^T$ matmul because $Q, K$ are full $[n, d]$.
- Longformer computes "only a fixed number of the diagonals of $QK^T$" — a *banded* matrix multiplication. Standard frameworks (PyTorch/TF) don't have a banded-matmul op.
- Solution: custom CUDA kernel built via TVM. Mentioned briefly; full details in Appendix A.

**Practical impact**: vanilla PyTorch implementations of SWA are slow because the banded-matmul has to be simulated with a masked dense matmul (which doesn't actually save FLOPs). Custom kernels (BlockSparse, TVM, later FlashAttention-with-mask) are required to realize the asymptotic gain.

## 5. Empirical ablations worth showing on a slide

1. **Figure 2** — the 4-panel attention-pattern comparison (Full vs SWA vs Dilated SWA vs Global+SWA). Canonical visual for the talk's (c) section.
2. **Figure 1** — runtime vs memory scaling chart. Linear (Longformer) vs quadratic (dense). Strongest one-glance evidence.
3. **Receptive-field-stacking math** ($\ell \cdot w$): why SWA at $w = 512$ per layer can see 12K tokens at depth 24.
4. **Per-layer variable window size** (§4.1): small $w$ low, large $w$ high. This is the design that became Gemma's local-global interleaving.

## 6. Where it shipped + how production differs from paper

### Modern open-frontier descendants (the actual reason this paper matters)

The paper itself (2020) targets RoBERTa-style encoder pretraining + downstream NLP tasks. The intellectual descendants in 2023-2026 frontier decoder-only LLMs:

- **Mistral 7B (2023)**: uses fixed SWA throughout. Window size 4096; ~16K context via stacked-receptive-field math.
- **Gemma 2 (2024)**: interleaves SWA layers (window 4096) with full-attention layers in a 1:1 ratio.
- **Gemma 3 (2025)**: same idea, **5:1 local:global** with window 1024.
- **Llama 4 (2026)**: iRoPE pattern — interleaves chunked-RoPE local attention with NoPE global layers.
- **Cohere Command A (2025)**: SWA-RoPE blocks interleaved 3:1 with Full-NoPE.

**These adoption claims need direct verification against each model's tech report before going on a slide.** Plan.md research-agent identified them; I haven't read those papers in this session.

### What's different in production

- Longformer's "global attention on task-specific tokens" pattern (§2c) didn't transfer — decoder-only generation doesn't have a `[CLS]` analog. Instead, frontier models use **periodic full-attention layers** (Gemma's 1:1, Gemma 3's 5:1, Llama 4's iRoPE).
- The "dilated SWA" pattern (§2b) didn't survive. The single biggest descendant is plain SWA + periodic full-attention layers.
- Custom TVM kernel was eventually subsumed by FlashAttention-style block-sparse masked attention.

---

## Talk-relevant takeaways

- **SWA's core insight is "stacked layers compound receptive field"**: a window of $w$ at every layer, $\ell$ layers deep, lets the model see $\ell \cdot w$ tokens — *without* paying $O(n^2)$ per layer. This is the load-bearing argument for SWA's place in the talk.
- **The receptive-field math doesn't free us from full attention entirely.** All frontier descendants interleave SWA with some flavor of full attention. Gemma 2: 1:1. Gemma 3: 5:1. Llama 4: ratio buried in their report. The "fully SWA" approach Longformer originally proposed is *not* what frontier ships — it's SWA + periodic full attention. Worth highlighting.
- **SWA attacks prefill compute, NOT KV cache size.** Each token still needs its KV stored; we're just computing fewer dot products per query. This is the "complementary to (a)" framing: KV compression and sparse attention attack different walls.
- **Longformer is the *right* anchor for the (c) section's static-sparse half**, replacing Sparse Transformer. Reasons: (1) text-domain native; (2) actually shipped in frontier LLMs (via descendants); (3) Figure 2 is cleaner than Sparse Transformer's Figure 3 for the audience; (4) the "stacked receptive field" math is more transferable.

## Things I did NOT verify in this session

- Specific window-size choices for Mistral 7B, Gemma 2/3, Llama 4, Command A — claimed from plan.md research-agent reports. Need direct verification before any specific numbers go on a slide.
- The exact local:global interleave ratios — Gemma 2 (1:1), Gemma 3 (5:1), Llama 4 iRoPE ratio, Command A (3:1) — all from plan.md, not from reading those papers this session.
- Whether FlashAttention v2/v3 implementations make Longformer's custom kernel obsolete. Training-time claim.

## Slide implications

For the (c) section, **Longformer gets a real slide**:

- Title: "Sliding window attention — let depth do the long-range work"
- Visual: Figure 2 (or a simplified 3-panel: full / SWA / SWA+global)
- Key bullets:
  - SWA: each token attends to $w$ neighbors. $O(n \cdot w)$ instead of $O(n^2)$.
  - Receptive field at depth $\ell$: $\ell \cdot w$. So 24 layers × $w = 512$ → 12K-token effective context.
  - Linear memory and time.
- A second beat (same slide or next): adoption in modern frontier — **Mistral 7B, Gemma 2 (1:1), Gemma 3 (5:1), Llama 4 (iRoPE)** — all use SWA + periodic full attention. Production didn't take Longformer's "SWA everywhere" approach.
- Setup for next slide: this is still STATIC sparsity. The patterns are fixed at design time. Modern trainable sparse attention (NSA, DSA) lets the model learn which positions to attend to. → c8.

If the adoption section gets long, split into two slides: one for Longformer-the-mechanism, one for "what frontier did with it." But initial cut at one slide.
