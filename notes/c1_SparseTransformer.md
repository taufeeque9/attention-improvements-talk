# c1 — Sparse Transformer

**Paper**: Child, Gray, Radford, Sutskever (OpenAI), *Generating Long Sequences with Sparse Transformers*, April 2019. [arXiv:1904.10509](https://arxiv.org/abs/1904.10509). [PDF](../papers/c1_SparseTransformer_Child2019.pdf).
**Length**: 10 pages, read in full this session.
**Why it matters**: The **first sub-quadratic attention pattern that frontier-shipped.** GPT-3 used a direct descendant of this. Opens the (c) section as the historical anchor.

---

## 1. Bottleneck targeted

**Prefill compute** — $O(n^2)$ attention scaling. Authors want to model sequences of length 12,288 (Enwik8 text), 65,536 (ImageNet pixel-level), and up to **1,048,576** (classical audio at 12 kHz). Dense attention is infeasible at these lengths.

The paper's stated goal: "self-attention to model sequences of length one million or more" (abstract).

## 2. Mechanism — factorized attention

**Core idea**: split the dense $n \times n$ attention matrix into $p$ steps. Each step's per-position subset has size $|A_i^{(m)}| \propto n^{1/p}$. Composing $p$ steps gives every output position a path to every input position. Total cost: $O(n \cdot n^{1/p})$. The paper uses $p = 2 \Rightarrow O(n \sqrt{n})$.

**Two factorization patterns** (Figure 3):

### Strided pattern
- Head 1: attends to the last $l$ positions ("local window"). $A_i^{(1)} = \{i-l, \dots, i\}$.
- Head 2: attends to every $l$-th position from earlier. $A_i^{(2)} = \{j : (i-j) \bmod l = 0\}$.
- $l \approx \sqrt{n}$. For $n = 1024$, $l = 32$.
- **Best for**: data with periodic structure (images at fixed width, music at beat tempo).

### Fixed pattern
- Head 1: same local window as before.
- Head 2: attends to specific "summary" positions — the last $c$ positions of each block of size $l$. $A_i^{(2)} = \{j : \lfloor j/l \rfloor = \lfloor i/l \rfloor\}$ plus summary cells.
- $l \in \{128, 256\}$, $c \in \{8, 16, 32\}$ — chosen empirically.
- **Best for**: data without spatial periodicity (text). The summary cells provide each block with a small fixed-size global context.

### How factorized heads are composed (§5.1)
Three options:
1. Alternate factorized head types per residual block: layer 1 uses $A^{(1)}$, layer 2 uses $A^{(2)}$, etc.
2. Merged head: a single attention over the union $A^{(1)} \cup A^{(2)}$ (constant-factor more cost).
3. Multi-head: each of $n_h$ heads runs its own factorized attention in parallel.

GPT-3 went with option 1 (alternating dense and sparse layers — see §6 below).

## 3. Cost reduction — claimed and measured

**Asymptotic**: $O(n^2) \to O(n \sqrt{n})$. For $n = 12{,}288$: $n/\sqrt{n} = 111$, so each attention step touches $\sim 111$ positions instead of $\sim 12{,}288$ — **~110× less compute per attention step**.

**Measured wall-clock** (Table 2):

| Setting | Time/iter |
|---|---|
| Enwik8 ($n = 12{,}288$), dense | 1.31 |
| Enwik8, fixed sparse | 0.55 (**~2.4× faster**) |
| Enwik8, strided sparse | 0.35 (**~3.7× faster**) |
| CIFAR-10 ($n = 3{,}072$), dense | 0.54 |
| CIFAR-10, strided | 0.38 (**~1.4× faster**) |

Speedup grows with $n$ (as expected from $\sqrt{n}/n$ ratio).

**Quality** (Table 1, bits-per-dim density modeling — lower is better):

| Task | Sparse Transformer | Prior SOTA |
|---|---|---|
| CIFAR-10 (3K context) | 2.80 | 2.85 (PixelSNAIL) |
| Enwik8 (12K context) | 0.99 | 0.99 (Transformer-XL 277M, but Sparse uses 95M) |
| ImageNet 64×64 (12K context) | 3.44 | 3.52 (SPN 150M) |

The headline: sparse attention **matches dense** at fixed parameter budget AND **trains faster** AND **scales to 1M+ contexts** in some regimes.

## 4. Hardware-aware aspects

§5.5 "Efficient block-sparse attention kernels":

- Custom GPU kernels for the two sparsity patterns.
- Local windows: direct sub-block matmul of slice-out queries, keys, values.
- Strided: transpose the keys matrix, then local-window kernel handles it.
- Fixed: aggregate the summary positions, then block-compute.
- Fused softmax kernel with register-level data reuse.
- Removed the upper-triangular bias terms entirely (causal mask is structural to the slicing).

**Mixed-precision (§5.6)**: weights in FP32, activations and gradients in FP16, dynamic loss scaling. Standard mid-2019 Volta-V100 setup, but mentioned because it was a precondition for training the 128-layer models.

## 5. Empirical ablations worth showing on a slide

1. **Figure 3** — the canonical 6-panel diagram: full attention vs strided sparse vs fixed sparse, both showing the receptive-field pattern AND the resulting connectivity matrix. This is THE visual for the (c)-section opener.
2. **Time/iter table** (Table 2) — concrete 2.4–3.7× speedup at $n = 12{,}288$.
3. **Up to 1,048,576 sequence length** (Table 4) — proof-of-concept claim that audio at $\sim$1M tokens is feasible.
4. **Figure 2** — the learned-attention-pattern visualizations from a dense 128-layer model on CIFAR-10, motivating that "real" attention IS sparse, so a sparsity prior doesn't sacrifice much.

## 6. Where it shipped + how production differs from paper

### GPT-3 adoption

GPT-3 (Brown et al. 2020) §2.1 reportedly describes its sparse-attention recipe as something like:
> "we use alternating dense and locally banded sparse attention patterns in the layers of the transformer, similar to the Sparse Transformer (Child et al., 2019)"

**Caveat**: I'm quoting from training-time memory. We do not have the GPT-3 paper downloaded in `papers/`. Before this quote goes on a slide, re-verify against arXiv 2005.14165 §2.1 directly. The "alternating dense + locally banded sparse" framing is consistent with option 1 from §5.1 of the Sparse Transformer paper.

If the quote checks out: **GPT-3 = first frontier deployment of sub-quadratic attention.**

### What's different in production

- The static patterns (strided or fixed) chosen *a priori* — no learning, no input-dependent selection. This is the major limitation that motivates modern *learned* sparse attention (NSA, DSA, MoBA — c8/c9/c10).
- Most subsequent open-frontier work moved to Longformer-style sliding window (c2 → Mistral / Gemma / etc.) rather than Sparse Transformer's specific patterns. The strided pattern survived in image/audio domains; text generation didn't keep it after GPT-3.
- The custom block-sparse kernels were eventually subsumed by FlashAttention v2/v3 which support sparse masking patterns natively.

---

## Talk-relevant takeaways

- **Sub-quadratic attention has been in production since GPT-3 (2020).** The "sparse attention is new" framing of NSA/DSA/MoBA is wrong; what's new is *learned* sparse attention. The Sparse Transformer's static pattern was the proof-of-concept.
- **The $O(n \sqrt{n})$ idea** generalizes: you can factor attention into $p$ steps with subset size $n^{1/p}$. Most modern sparse attention is still $p = 2$ (a local window + a long-range selector), just with the selector being learned rather than fixed.
- **Strided vs fixed** is an early articulation of "local + global" patterns. Longformer (c2) makes this distinction cleaner; Gemma's interleaving formalizes it. We can frame the lineage: Sparse Transformer (static, factor 2) → Longformer (static + global tokens) → trainable sparse (NSA / DSA / MoBA) → V4 hybrid (CSA + HCA).
- **Best slide visual**: Figure 3, full-attention vs strided vs fixed. Hard to beat as a one-glance demonstration of "we don't need all $n^2$ entries."
- **Connection to talk's per-step framework**: sparse attention attacks the *prefill* wall ($O(n^2)$ compute). It does NOT help decode arithmetic intensity directly (you still load the cache; you just have fewer values to attend to per query — but the load is the bandwidth-bound part). So sparse attention complements GQA/MLA, not replaces.

## Things I did NOT verify in this session

- That GPT-3's exact pattern matches the strided variant. The GPT-3 paper says "similar to," not "identical." Worth flagging on slides — production probably used a tweaked variant.
- That FlashAttention v2/v3 natively support Sparse Transformer's patterns. My claim from training; not verified by reading FlashAttention papers in this session.
- The specific 1M-token audio result — paper says "up to 1,048,576 timesteps long, albeit with extremely few parameters (3 million)." So the long-context claim is real but at toy model size. Worth flagging if the headline shows up on a slide.

## Slide implications

**Decision: drop from slides** (user, May 2026). Reasoning: paper's center of gravity is image/audio generation; modern text-LLM sparse-attention lineage goes through Longformer (c2) and its descendants (Mistral SWA, Gemma local-global interleave), not directly through Sparse Transformer.

GPT-3 used "something similar" — that's worth a single sentence in the speaker notes when introducing the (c) section (*"sub-quadratic attention has been deployed since GPT-3"*) — but not a full slide.

Notes file retained as historical reference. If we want to reinstate a half-slide later (e.g. "sparse attention has been in production since 2020"), the visual would still be Figure 3.
