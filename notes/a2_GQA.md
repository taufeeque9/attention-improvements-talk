# a2 — Grouped-Query Attention (GQA)

**Paper**: Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai (Google Research), *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245). [PDF](../papers/a2_GQA_Ainslie2023.pdf).
**Length**: 7 pages (5 main + refs + 1-page appendix). Read in full.

---

## 1. Bottleneck targeted

**Same as MQA**: autoregressive decoder inference is bandwidth-bound by reloading the KV cache each step (§1, citing Shazeer 2019 + Pope et al. 2022). GQA inherits MQA's framing wholesale and asks: **how much KV sharing can we afford without paying the quality / stability cost?**

## 2. Mechanism — exact change vs. MHA and MQA

**GQA-$G$**: split $H$ query heads into $G$ groups; each group shares one KV head pair.

- $G = 1$: GQA-1 = MQA (one shared KV pair).
- $G = H$: GQA-$H$ = MHA (one KV pair per head).
- $1 < G < H$: the interpolation.

The arithmetic effect on the cache:

| | MHA | GQA-$G$ | MQA (= GQA-1) |
|---|---|---|---|
| KV-head count | $H$ | $G$ | $1$ |
| Cache size | $\propto H$ | $\propto G$ | $\propto 1$ |
| Cache shrink factor vs MHA | 1× | $H/G$ | $H$ |

**Choice of $G$ in production**: paper picks $G = 8$ ("a favorable middle ground", §3.3). No theoretical derivation — purely empirical, based on Figure 6's time-per-sample curve showing the knee at $G = 8$.

**Critical scoping detail**: GQA is applied only to **decoder self-attention and cross-attention**, NOT encoder self-attention (§2.2). Encoder is sequence-parallel → already compute-bound → no bandwidth gain from sharing. For decoder-only models (Llama, Mistral, etc., which the paper is *not* about but everyone applies it to), this constraint is moot.

### Uptraining recipe — the second contribution

Don't train GQA from scratch — convert an existing MHA checkpoint:

1. **Convert**: mean-pool the $H$ KV heads down to $G$ group KV heads. Mean-pool beats "pick the first head" and beats "random init" (Figure 4).
2. **Uptrain**: continue pre-training on the original recipe for $\alpha = 5\%$ of original training steps. ~600 TPUv3 chip-days for T5-XXL.
3. Quality after $\alpha = 0.05$ is essentially saturated; gains beyond $\alpha = 0.1$ are negligible (Figure 5).

Interesting detail: **GQA already works decently at $\alpha = 0$ (no uptraining at all)**, while MQA needs uptraining to be useful at all (Figure 5). Mean-pooling preserves more of the original capacity.

## 3. Cost reduction — claimed and measured

**Table 1, T5-XXL** on summarization (CNN/DM, arXiv, PubMed, MediaSum, MultiNews) + WMT translation + TriviaQA QA:

| Model | $T_\text{infer}$ (s/sample) | Avg quality |
|---|---|---|
| MHA-Large | 0.37 | 46.0 |
| **MHA-XXL** | **1.51** | **47.2** |
| **MQA-XXL** (uptrained) | **0.24** | 46.6 |
| **GQA-8-XXL** (uptrained) | **0.28** | **47.1** |

Reading this row by row:

- MHA-XXL → MQA-XXL: 6.3× faster, –0.6 avg quality.
- MHA-XXL → **GQA-8-XXL: 5.4× faster, only –0.1 avg quality**.
- GQA-XXL is **both faster AND higher quality than MHA-Large** (0.28 s vs 0.37 s; 47.1 vs 46.0) — a strict Pareto win that recovers more than the parameter difference would suggest.

**Figure 3** is the punchline plot: a 2D scatter of performance vs inference time. GQA sits in the empty top-left corner, dominating both MHA points (Large and XXL).

## 4. Hardware-aware aspects

Two big points, both in §2.2:

1. **KV cache scales with $G$, not $H$.** Cache size $\propto bGmd_k$, so $G$ controls bandwidth.
2. **Sharding matters at scale.** Standard tensor-parallel sharding (Pope et al. 2022) **replicates** the single MQA KV head across all $N$ model partitions — wasted memory and compute. GQA with $G = N$ matches the number of partitions exactly, so each partition gets its own KV head — **no replication waste**. This is the underrated reason GQA beats MQA at frontier scale: MQA's "one KV head" doesn't shard nicely.

The paper also notes (§2.2): "larger models suffer relatively less from memory bandwidth overhead from attention, as the KV-cache scales with model dimension while model FLOPs and parameters scale with the square of model dimension." So bandwidth pressure goes down with scale anyway — GQA is the right knob because MQA's quality cost becomes harder to justify when the bandwidth gain is smaller.

## 5. Empirical ablations worth showing on a slide

1. **Figure 2** — the visual structure diagram of MHA / GQA / MQA (queries fanning into groups of shared KV). The canonical visual for the (a) section.
2. **Figure 3** — Pareto plot (time vs quality), showing GQA-8 dominates MHA-Large.
3. **Figure 6** — time-per-sample vs $G$ (1, 4, 8, 16, 32, 64). The curve is flat from $G=1$ to $G=8$, then knees up sharply. Justifies $G = 8$.
4. **Table 1** — concrete numbers (T_infer 1.51s → 0.28s, quality 47.2 → 47.1).
5. **Appendix A** — training-stability paragraph. Crucial for the talk's narrative ("MQA's instability is what made GQA necessary").

## 6. Where it shipped + how production differs from paper

**Paper's own scope**: T5-XXL only (encoder-decoder). The paper itself doesn't claim adoption beyond T5; it explicitly notes (§1) that as of 2023 "LLaMA" doesn't use MQA, and (Limitations) that they expect "GQA to have a stronger advantage over MQA" for decoder-only models without verifying.

**Downstream adoption** (per plan.md, *not* verified by reading other papers this session): Llama 2/3/3.1/4, Mistral 7B, Mixtral, Qwen 2.5/3, Gemma 2/3, Phi-3/4, Command A. **Will verify against the Llama 3 herd paper when we hit it.**

**Production differences from the paper**:
- The paper uses encoder-decoder T5; production uses decoder-only.
- Production trains from scratch with GQA (no "uptrain from MHA" step needed — by the time GQA was published, no one was training MHA at scale anymore).
- Most production models use $G = 8$ regardless of $H$ (Llama 3 8B: $H=32, G=8$; Llama 3 70B: $H=64, G=8$; Mistral 7B: $H=32, G=8$ — *to verify when reading those reports*).

---

## Confirmations / corrections of my earlier notes

- **CONFIRMED** (Appendix A, verbatim from the paper):

  > "We find that multi-query attention can lead to training instability during fine-tuning, in particular combined with long input tasks. We trained multiple T5-Large models with multi-query attention from scratch. In each case, pre-training suffered from frequent loss spikes and the final models diverged immediately when fine-tuning on long-input tasks. Uptrained multi-query attention models are more stable but still display high variance ... Uptrained grouped-query attention models, however, appear to be stable, so we did not investigate further on the root causes of multi-query instability."

  My MQA notes claim "At billion-param scale: training instability and larger quality drops than the WMT14 numbers suggest. (reported in the GQA paper as motivation — we'll verify next.)" — **this is now verified**.

- **The paper does NOT theoretically derive $G = 8$**. It's empirical. I should NOT claim there's a principled reason for the choice — Figure 6's knee is the actual evidence.

- **GQA was independently developed by Rabe 2023** (Markus Rabe's flaxformer implementation, cited in §4). Worth a line on slides if we want to give credit beyond Ainslie et al.

## Talk-relevant takeaways

- **GQA's pitch is "interpolation knob"** between MHA and MQA — pick where on the Pareto curve you want to land. This framing is what makes it the canonical successor to MQA in the talk's KV-compression arc.
- **The hardware-aware story is the underrated half of the paper**: MQA's single KV head wastes sharding partitions; GQA fixes this. This argument scales better with model size than the per-step bandwidth argument does. Worth highlighting on a slide.
- **Uptraining is the practical knob**: at $\alpha = 5\%$, you get a model that's almost-MHA-quality at almost-MQA-speed. This is what made GQA *adoptable* — labs could retrofit existing checkpoints rather than retrain from scratch.
- **Best slide visual**: combine Figure 2 (structure) and Figure 3 (Pareto) on a single slide — the diagram for *what* and the curve for *why*.

## Todo

- Check models that use GQA use what values of heads and groups
- That GQA gives a "particularly good trade-off for larger models" as the paper claims (§2.2). The paper only evaluates up to T5-XXL (~11B), so this is extrapolation. Decoder-only Llama 70B etc. would be the real evidence — but downstream papers don't typically run the MHA-vs-GQA ablation at 70B because no one trains MHA-70B.
