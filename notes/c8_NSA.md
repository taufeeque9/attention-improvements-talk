# c8 — NSA (Native Sparse Attention)

**Paper**: Yuan, Gao, Dai, Luo, Zhao, Zhang, Xie, Wei, Wang, Xiao, Wang, Ruan, Zhang, Liang, Zeng (DeepSeek-AI + Peking University + UW), *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*, Feb 2025. [arXiv:2502.11089](https://arxiv.org/abs/2502.11089). [PDF](../papers/c8_NSA_DeepSeek.pdf).
**Length**: 25 pages. Read pp. 1-9 in this session (abstract + intro + critique of prior sparse + Methodology + Kernel design + start of Experiments). Skipped detailed experiments (§4.2-5) and ablations.
**Why it matters**: First **trainable-from-scratch ("native")** sparse attention. Direct predecessor to **DSA** (DeepSeek-V3.2, c9) — the production-shipped trainable sparse attention.

---

## 1. Bottleneck targeted

**Prefill compute** (and training cost for long context). $O(n^2)$ softmax attention costs 70-80% of total latency when decoding at 64K context (claimed §1). NSA wants to:
1. **Achieve real wall-clock speedup, not just theoretical FLOPs reduction** — most prior sparse attention failed at this.
2. **Train sparse from scratch** instead of bolting sparsity onto a dense pretrained checkpoint. This is the "native" in NSA.

§2 explicitly critiques prior sparse methods on two axes:
- **The illusion of efficient inference**: many methods get FLOPs reduction but no latency reduction, because (a) phase-restricted sparsity (one phase still dense), (b) incompatibility with GQA/MQA — head-independent selection forces loading the union of selections across heads, killing the GQA bandwidth win.
- **The myth of trainable sparsity**: most prior "trainable" sparse methods either use non-differentiable selection (k-means, SimHash → no gradient flow) or have token-granular selection that breaks FlashAttention-style blockwise memory access.

## 2. Mechanism — three parallel branches

Figure 2 is the architecture diagram: three branches (compression / selection / sliding window) with a learned gate combining their outputs.

For a query token $q_t$, each branch produces its own keys/values $(\tilde K_t^c, \tilde V_t^c)$ for $c \in \{\text{cmp}, \text{slc}, \text{win}\}$. Output:

$$
o_t^* = \sum_{c \in \{\text{cmp}, \text{slc}, \text{win}\}} g_t^c \cdot \text{Attn}(q_t, \tilde K_t^c, \tilde V_t^c)
$$

where $g_t^c$ are sigmoid gates from an MLP on the input.

### 2a. Compression branch (cmp) — coarse-grained

Aggregate consecutive blocks of keys/values into a single compressed representation:

$$
\tilde K_t^{\text{cmp}} = \{\varphi(k_{id+1 : id+l}) \,|\, 0 \le i \le \lfloor (t-l)/d \rfloor\}
$$

- Block length $l$, stride $d < l$ (overlapping blocks reduce information fragmentation).
- $\varphi$ is a learnable MLP with intra-block position encoding that maps $l$ keys → 1 compressed key.
- Sequence length collapses from $n$ to ~$n/d$.

### 2b. Selection branch (slc) — fine-grained, top-k blocks

**The key novelty**: top-$n$ block selection driven by the compressed attention scores.

1. Compute $p_t^{\text{cmp}} = \text{softmax}(q_t^\top \tilde K_t^{\text{cmp}})$ — already computed for the cmp branch, reused for free.
2. Convert compression scores to **selection-block scores** $p_t^{\text{slc}}$ via eq 9 (handles the case where selection blocks differ from compression blocks).
3. **Crucial for GQA**: aggregate scores across all heads in the GQA group:
   $$p_t^{\text{slc}\,'} = \sum_{h=1}^{H_\text{group}} p_t^{\text{slc}\,(h)}$$
   So all heads in a GQA group select the **same** top-$n$ blocks. This makes NSA compatible with GQA's KV-sharing.
4. Keep blocks with the top-$n$ scores; concatenate their (original, uncompressed) keys/values.

### 2c. Sliding window branch (win) — local

The most recent $w$ tokens, uncompressed. Standard SWA branch.

### Why three branches?

§3.3.3 says: "local patterns typically adapt faster and can dominate the learning process." Without a dedicated sliding-window branch, the compression/selection branches end up wasting capacity on local context. By separating, each branch can specialize. **Each branch has its own independent K, V projections** — prevents gradient interference.

Total active context per query: $N_t = \text{size}(\tilde K^{\text{cmp}}) + n \cdot \text{block-size}_\text{slc} + w$, kept $\ll t$.

## 3. Cost reduction — claimed and measured

**Figure 1 headline numbers** at 64K context length (vs Full Attention baseline, same 27B model):

| Phase | Speedup |
|---|---|
| **Decode** | **11.6×** |
| **Forward (prefill / training)** | **9.0×** |
| **Backward (training)** | **6.0×** |

The decode speedup is largest because that's the most bandwidth-bound phase (re-streams the full KV cache); NSA effectively loads $\ll n$ blocks per query.

**Quality** (also Figure 1):

| Benchmark | Full Attention | NSA |
|---|---|---|
| General avg | ~0.45 | **~0.47** |
| LongBench | ~0.43 | **~0.47** |
| Reasoning | ~0.09 | **~0.15** |

NSA *exceeds* Full Attention. This is the strongest part of the paper's claim — sparse attention can be *better* than dense, not just cheaper. Plausible mechanism: sparsity is a useful inductive bias that prevents the attention from getting distracted.

(I read these visually from Figure 1; precise numbers would require reading the experiment tables in §4.2 which I skipped.)

## 4. Hardware-aware aspects

§3.4 — explicitly designed for Tensor Core utilization.

**Triton kernels** with three key design choices (Figure 3):

1. **Group-Centric Data Loading**: load all query heads in a GQA group together at each position. Since they share the same sparse block selection, they share KV loads.
2. **Shared KV Fetching**: in the inner loop, sequentially load contiguous KV blocks indexed by the selection set. Block size aligned to selection block size.
3. **Outer Loop on Grid**: query/output loops in Triton's grid scheduler, simplifying parallelism.

The key claim: "near-optimal arithmetic intensity" by **eliminating redundant KV transfers through group-wise sharing** — exactly the GQA-compatibility argument from §2.

This is what differentiates NSA from prior sparse attention that *theoretically* reduces FLOPs but doesn't deliver wall-clock speedup. The kernel design IS the contribution as much as the algorithm.

## 5. Empirical ablations worth showing on a slide

1. **Figure 2** — the 3-branch architecture diagram with the compressed / selected / sliding attention masks side by side. Best single visual.
2. **Figure 1 (right panel)** — the bar chart: 11.6× / 9.0× / 6.0× speedups at 64K context. Concrete punchline.
3. **Figure 1 (left panel)** — performance comparison: NSA matches or beats Full Attention on three benchmark categories. Counterintuitive enough to be memorable.
4. **The GQA-compatibility argument** — a one-liner about why per-head selection breaks GQA bandwidth, and why NSA's group-shared selection fixes it. Connects (c) sparse to (a) GQA.

## 6. Where it shipped + how production differs from paper

### Production status

NSA itself was a **research paper from DeepSeek-AI**, not a shipped model. The architecture was demonstrated on a 27B-param backbone trained for 260B tokens — small by frontier standards.

**What actually shipped**: **DSA (DeepSeek Sparse Attention)** in DeepSeek-V3.2 — c9, which we'll read next. DSA is the production refinement of NSA, replacing the 3-branch architecture with the "lightning indexer" we saw briefly in V4's CSA. The V4 paper (c13) explicitly cites NSA as part of the lineage.

### Connections to other parts of the talk

- **DSA (c9)** = production trainable sparse attention shipped in V3.2. Originally framed (by me) as an "NSA descendant," but on closer reading DSA and NSA are **probably parallel research threads** at DeepSeek-AI — DSA cites NSA only once (for GQA-compatibility) and uses a structurally different design (1 branch + lightning indexer vs NSA's 3 branches). Likely different deployment targets: DSA for continued-PT-from-dense retrofit, NSA for from-scratch training. See c9 notes for the parallel-threads framing.
- **V4 (c13)** explicitly credits DSA (V4 §1: *"CSA ... performs DeepSeek Sparse Attention (DSA)"*). V4 paper does NOT mention NSA, despite 8+ NSA authors appearing in V4's author list and V4 CSA's 3-component design (compression + selection + sliding window) being structurally similar to NSA. NSA likely informed V4 implicitly through shared authors; not paper-credited.
- **MoBA (c10)** is a parallel approach by Moonshot AI, also trainable sparse — but uses block-level routing rather than NSA's compression+selection. Same era.

### What's different in production

- NSA's 27B / 260B-token training is a *research validation*, not a deployment. DeepSeek-V3.2 (frontier-scale; exact param count and training tokens to verify when reading c9) is what proved this approach at frontier scale.
- The Triton kernels in NSA are educational; production uses optimized CUDA / DSA-specific kernels.
- DSA simplifies NSA's 3-branch design — the V3.2 paper just has one "lightning indexer + selection" path on top of MLA. We'll see this when reading c9.

---

## Talk-relevant takeaways

- **"Native" = trainable from scratch.** The single most important thing about NSA. Prior sparse attention was bolted onto pretrained dense models and lost quality; NSA shows that if you train the model *with* the sparsity from the start, it can match or beat dense.
- **The 3 branches are pedagogically clean** — compression for coarse, selection for fine-grained, sliding for local. Each addresses a different "scale" of context. Inherits Longformer's local+global insight + adds a learned selection mechanism.
- **The GQA-compatibility argument is the underrated piece.** Sparse attention that selects different tokens per head defeats GQA's bandwidth win. NSA's group-shared block selection is what makes it production-viable on GQA-architecture models. This insight transfers directly to DSA and V4.
- **Hardware aligned > theoretically efficient.** The 11.6× decode speedup is real wall-clock, not just FLOPs reduction. Worth contrasting with earlier sparse attention "theoretical complexity" claims that didn't translate to speed.
- **Best slide visual**: Figure 2 (3-branch architecture). Concrete and self-explanatory.
- **Best punchline number**: 11.6× decode speedup at 64K context, matching/beating Full Attention quality.

## Things I did NOT verify in this session

- The selection block size $l'$ — paper doesn't explicitly state in §4.1. The "2560 tokens activated at 32K" measurement (§4.2) is the closest direct number.
- The ablation studies in §5 (relative contribution of each branch, sensitivity to block sizes, etc.). Could be useful if we want to argue specific design choices.
- Comparison with H2O, MInference, Quest, etc. — NSA's competitor sparse-attention methods that §4.3 benchmarks against (Table 2 has the numbers).

## Verified in §4 (refresh from this turn)

- **Hyperparameters** (§4.1): $l = 32$ (compression block), $d = 16$ (sliding stride), $n = 16$ (selected blocks, including 1 fixed initial + 2 fixed local), $w = 512$ (sliding window).
- **Pretraining context** (§4.1): "pretrained on **270B tokens of 8k-length texts**, followed by continued training and supervised fine-tuning on **32k-length texts** with YaRN to achieve long-context adaptation." Important context: NSA's headline "from scratch" pretraining was at 8K, not at longer context. The 32K evaluation numbers came after continued training.
- **Sparsity** (§4.2): "the average number of tokens activated in NSA when handling 32k sequence lengths" = **2560 tokens** = ~8% of context.
- **Table 1** (general benchmarks): NSA average **0.456** vs Full Attention **0.443**. NSA wins on 7 of 9 metrics.
- **Table 2** (LongBench): NSA average **0.469** vs Full Attention **0.437**. NSA wins on most subsets. Notable +0.087 / +0.051 on multi-hop QA (HPQ, 2Wiki).
- **Figure 5** (Needle-in-a-Haystack at 64K): NSA achieves perfect retrieval across all depths.

## Slide implications

For the (c) section, NSA should get its own slide as the **canonical "trainable sparse attention" reference**. Suggested structure:

- Title: "Trainable sparse attention — NSA's 3-branch architecture"
- Figure 2 (the 3-branch + attention masks visual)
- Three bullets, one per branch (cmp / slc / win)
- Punchline: 11.6× / 9.0× / 6.0× speedups at 64K, matching Full Attention quality
- GQA-compatibility one-liner: "group-shared block selection is what makes this work with shared KV"
- Setup for c9: "shipped in DeepSeek V3.2 as DSA, simplified to a single 'lightning indexer' path → next slide"

One slide should be enough. If we want NSA + DSA as two slides, NSA carries the trainable-sparse architecture story; DSA carries the production-shipped headline + the simplification.
