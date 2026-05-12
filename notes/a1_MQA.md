# a1 — Multi-Query Attention (MQA)

**Paper**: Noam Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, 2019. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150). [PDF](../papers/a1_MQA_Shazeer2019.pdf).
**Length**: 9 pages, single-author, very concise.

---

## 1. Bottleneck targeted

**Decode-time memory bandwidth, not compute.**

In autoregressive inference, queries are generated one position at a time — no sequence parallelism. The per-step cost is dominated by *reloading the K and V tensors from HBM* to compute attention against all previous positions. Shazeer's framing:

- Modern GPUs/TPUs have **compute capacity ~100× higher than memory bandwidth**. A kernel is fast only when its arithmetic-intensity (FLOPs per byte loaded) is high.
- For batched MHA decode, the ratio of memory-access to arithmetic operations is **Θ(n/d + 1/b)** (paper §2.4.1). When `n ≈ d` (long context) or `b ≈ 1` (small batch), this ratio approaches 1 — and the kernel becomes bandwidth-bound.

The `n/d` term is what makes long-context decode slow. It comes from the bhmk size of the cached K (and bhmv of V) — i.e., the per-step cost of re-streaming the entire KV cache from HBM.

## 2. Mechanism — exact change vs. MHA

**Strip the heads dimension from K and V (and their projections). Keep it on Q.**

| Tensor | MHA shape | MQA shape |
|---|---|---|
| `P_q` (query proj) | `[h, d, k]` | `[h, d, k]` (unchanged) |
| `P_k` (key proj) | `[h, d, k]` | `[d, k]` |
| `P_v` (value proj) | `[h, d, v]` | `[d, v]` |
| `P_o` (output proj) | `[h, d, v]` | `[h, d, v]` (unchanged) |
| Cached K | `[b, h, m, k]` | `[b, m, k]` |
| Cached V | `[b, h, m, v]` | `[b, m, v]` |

The einsum that produces K and V loses one index:
- MHA: `K = einsum("bmd, hdk -> bhmk", M, P_k)`
- MQA: `K = einsum("bmd,  dk -> bmk",  M, P_k)`

Logits/output einsums broadcast the shared K, V across all `h` query heads:
- MHA: `logits = einsum("bhnk, bhmk -> bhnm", Q, K)`
- MQA: `logits = einsum("bhnk, bmk  -> bhnm", Q, K)`

Conceptually: **all h query heads ask different questions; they all consult the same single key/value memory.** "One write-head."

To keep total param-count matched in experiments, Shazeer widens the FFN (d_ff 4096 → 5440 in the WMT model). So MQA effectively *moves parameter budget from KV projections into FFN.*

## 3. Cost reduction — claimed and measured

**Asymptotic**: decode mem:compute ratio reduces from Θ(n/d + 1/b) → **Θ(1/d + n/(d·h) + 1/b)**. The dominant `n/d` term shrinks by a factor of **h** (number of heads).

**KV cache size**: shrinks by factor of h. With h=8, that's an 8× reduction in cached bytes per layer.

**Measured wall-clock** (WMT14 EN-DE, TPUv2, batch 1024, seq len 128, h=8):

| Phase | MHA | MQA | Speedup |
|---|---|---|---|
| Training (per token) | 13.2µs | 13.0µs | 1.02× (essentially free) |
| Encoder inference (per token) | 1.7µs | 1.5µs | 1.13× |
| **Decoder inference (per token)** | **46µs** | **3.8µs** | **~12.1×** |
| Beam-4 decode (per token) | 203µs | 32µs | ~6.3× |

The decoder speedup is the headline. Training cost is essentially unchanged — MQA is an inference optimization that is roughly free at train time.

**Quality cost**:
- WMT14 dev BLEU: 26.7 → 26.5 (–0.2)
- WMT14 dev ln(PPL): 1.424 → 1.439 (+0.015)
- WMT14 test BLEU beam-4: 28.4 → 28.5 (MQA actually *better* on test by 0.1)
- Billion-Word LM dev PPL: 29.9 → 30.2 (+0.3)

**Importantly**, MQA dominates the obvious alternative — reducing `h` (number of heads) — on a param-matched basis:
- h=8, MHA: BLEU 26.7
- h=2 MHA, larger FFN: BLEU 26.2
- h=1 MHA, larger FFN: BLEU 25.8
- h=8 MQA, larger FFN: **BLEU 26.5**

So MQA buys the bandwidth savings of "one KV head" without paying the quality hit of "one attention head."

## 4. Hardware-aware aspects

Yes — the paper is fundamentally a hardware argument:

- The §2.3.1 / §2.4.1 performance analyses are explicit arithmetic-intensity calculations.
- The motivation is the GPU/TPU FLOPs ≫ memory bandwidth gap.
- The optimization isn't reducing total FLOPs; it's reducing **memory traffic** so the same FLOPs can actually be retired.
- Training is parallel over sequence positions → high arithmetic intensity → not bandwidth-bound → MQA gives no training speedup, by design.

## 5. Empirical ablations worth showing on a slide

**Highest-signal numbers** (Table 1 + Table 2):

1. **Decoder per-token**: 46µs → 3.8µs (~12×) — the punchline.
2. **Param-matched alternatives table** (Table 1): MQA at h=8 ≫ MHA at reduced h. This makes the case that MQA isn't just "fewer heads."
3. **Orthogonality with local attention** (Table 2): local + MQA stacks → 3.3µs/token, both gains compose.
4. **Training cost unchanged** (Table 2): the 13.2µs → 13.0µs row is rhetorically important — "you pay nothing during training, you win ~12× at inference."

## 6. Where it shipped + how production differs from paper

**Reported adopters of pure MQA**: PaLM (Chowdhery et al. 2022), Falcon (TII), StarCoder. *Not directly verified in this conversation — claims sourced from research-agent web search. Verify by reading those model reports if precise attribution matters for slides.*

**What changed in production**:
- **Quality at scale was worse than the paper suggests.** At billion-parameter scale and longer training, MQA was found to be harder to train and to lose more quality than the WMT14 211M numbers indicated. This is *the* motivation for GQA — explicit in the GQA paper (Ainslie et al. 2023; will verify when reading a2).
- **GQA (g groups of shared K/V) is the in-between point** that recovers most of the quality while keeping most of the bandwidth wins.
- Most current frontier open models use GQA, not MQA.

**Open question to resolve when reading a2**: what's the exact quality-vs-bandwidth Pareto trade as you vary `g` from 1 (MQA) to h (MHA)? Ainslie reports this; will check.

---

## Talk-relevant takeaways

- **The framing of the entire (a) category lives in this paper.** "Memory bandwidth, not compute, is the decode bottleneck — and it gets worse with long context." Once the audience sees Θ(n/d + 1/b), the rest of the KV-compression arc (GQA → MLA) is almost mechanical.
- **Best visual**: the einsum-signature diff. `bmd,hdk→bhmk` vs `bmd,dk→bmk`. One letter. ~12× decoder speedup. That's the slide.
- **Best number**: 46µs → 3.8µs for the decoder. Even though the absolute numbers are 2019-era and small-model-era, the ratio is what matters and it carries over.
- **Cautionary**: MQA's BLEU drop of 0.2 on a 211M translation model massively undersells the quality risk at scale. Don't oversell stability.

## Things I noted but did NOT verify

- That PaLM / Falcon / StarCoder actually use MQA — relayed from prior research-agent output, not from those papers directly.
- That GQA's introduction was motivated by MQA's training instability at scale — my read of the field, to confirm against a2.
- That MQA was "largely superseded" — true based on the current open-frontier landscape, but I should not generalize beyond what I can cite. (Closed models may still use MQA somewhere.)
