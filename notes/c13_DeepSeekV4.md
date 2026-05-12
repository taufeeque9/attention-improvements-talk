# c13 — DeepSeek-V4 (Hybrid CSA + HCA Attention)

**Paper**: DeepSeek-AI, *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*, April 2026. [HF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro). [PDF](../papers/c13_DeepSeekV4.pdf).
**Length**: 58 pages. Notes below cover §1 (intro) and §2 (architecture, pp. 4-14). Did NOT read §3-6 (infrastructure, training, eval, post-training) in this session.

---

## 1. Bottleneck targeted

**Both prefill compute (O(n²)) and decode KV bandwidth, in the 1M-token regime.** §2.3:

> "As the context length reaches extreme scales, the attention mechanism emerges as the dominant computational bottleneck. … We design two efficient attention architectures — CSA and HCA — and employ their interleaved hybrid configuration."

Specifically, V4 attacks: (a) per-layer sequence length, by compressing $n$ tokens into $n/m$ or $n/m'$ compressed entries before attention runs; (b) per-query active-context size, by sparse-selecting top-$k$ of the compressed entries; (c) KV cache bytes, by mixed-precision storage (BF16 RoPE dims + FP8 elsewhere).

## 2. Mechanism — exact change vs. baseline

V4 replaces every transformer block's attention with **either CSA or HCA**, in an interleaved pattern. (Paper says "interleaved hybrid configuration" — exact ratio not stated in §2.3 of what I read; in [paper's open-source code](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference) is the unambiguous spec.)

### CSA (Compressed Sparse Attention) — Fig. 3

For input hidden states $H \in \mathbb{R}^{n \times d}$:

1. **Token-level compressor.** Compute two KV streams $C^a, C^b \in \mathbb{R}^{n \times c}$ and weights $Z^a, Z^b \in \mathbb{R}^{n \times c}$. Then every consecutive $m$ KV entries are softmax-pooled into a single compressed entry $C_i^\text{Comp} \in \mathbb{R}^c$. Overlapped windows (uses $2m$ entries per output). **Sequence length collapses to $n/m$.**
2. **Lightning Indexer.** Cheap query-side path: down-project query $h_t$ to a latent $c_t^Q \in \mathbb{R}^{d_c}$, then up-project to $n_h^I$ indexer-query heads $q_{t,h}^I$. Compute index score $I_{t,s} = \sum_h w_{t,h}^I \cdot \mathrm{ReLU}(q_{t,h}^I \cdot K_s^\text{IComp})$ between query $t$ and each compressed-block index $s$. **Same DSA trick as V3.2** (cited inline).
3. **Top-$k$ selector.** Keep only $C_s^\text{Comp}$ where $I_{t,s} \in \text{Top-k}(I_{t,:})$. The query token attends to exactly $k$ compressed entries.
4. **Sliding window branch.** Additionally keep the most recent $n_\text{win}$ uncompressed KV entries (for local fine-grained dependencies — recent tokens have higher relevance).
5. **Core attention: Shared-KV MQA.** From the same compressed latent $c_t^Q$, up-project to $n_h$ query heads $q_{t,i}$. Run multi-query attention over `concat(selected compressed entries, sliding-window entries)`. The compressed latent is **shared** with the indexer queries (efficient).
6. **Grouped Output Projection.** Split $n_h$ outputs into $g$ groups, project each group to $d_g$ dim, concatenate, project to $d$. Avoids the large $n_h \cdot c \to d$ output proj.

### HCA (Heavily Compressed Attention) — Fig. 4

1. **Token-level compressor**, same idea as CSA but with **$m' \gg m$** — much more aggressive compression. Non-overlapping windows. Sequence length collapses to $n/m'$.
2. **No sparse selection.** Each query attends densely to all $n/m'$ compressed entries.
3. **Sliding window branch** — same as CSA, adds $n_\text{win}$ recent uncompressed entries.
4. **Shared-KV MQA core + Grouped Output Projection** — same as CSA.

### Other details (§2.3.3)

- **Q/KV RMSNorm**: applied to each head of queries and to the (single) head of compressed KV entries, just before core attention. Replaces V3.2's QK-Clip technique; lets V4 use Muon optimizer without the clipping stability hack.
- **Partial RoPE**: only the **last 64 dims** of Q and KV entries get RoPE.
- **Attention sink** (StreamingLLM trick, cited to Xiao et al. 2024 and OpenAI 2025): adds learnable sink logits $z_h'$ to the softmax denominator. Lets each head adjust its total attention mass — possibly even near zero. Same in both CSA and HCA.

## 3. Cost reduction — claimed and measured

**vs DeepSeek-V3.2 at 1M-token context** (Fig. 1, also stated in §2.3.4):

| Model | Single-token inference FLOPs | KV cache |
|---|---|---|
| V4-Pro (1.6T / 49B active) | **27%** of V3.2 (3.7× lower) | **10%** of V3.2 (9.5× smaller) |
| V4-Flash (284B / 13B active) | **10%** of V3.2 (9.8× lower) | **7%** of V3.2 (13.7× smaller) |

**vs the common BF16 GQA-8 baseline** (head dim 128) at 1M-token context:

> "the KV cache size of DeepSeek-V4 series can be dramatically reduced to approximately 2% times of that baseline" (§2.3.4)

i.e. ~50× smaller KV cache than vanilla GQA-8 at 1M context.

The reductions come from three stacking effects:
1. Token-level compression by factor $m$ (CSA) or $m'$ (HCA).
2. Sparse top-$k$ selection further reduces the *active* context per query (CSA only).
3. Mixed-precision storage: BF16 for the 64 RoPE dims, FP8 for the rest → ~50% smaller KV bytes (§2.3.4).
4. Lightning indexer attention computed in **FP4** (extreme — only viable because the indexer is just for selection, not the value path).

## 4. Hardware-aware aspects

- **Mixed-precision KV** is explicit hardware accommodation: BF16 only where needed for RoPE rotation accuracy; FP8 elsewhere.
- **FP4 indexer attention** is bleeding-edge precision-aware design.
- **Grouped Output Projection** avoids the large output matmul that would otherwise dominate at long context.
- **On-disk KV cache** (§3.5.2, not read this session) is mentioned in the abstract — implies V4's KV is so reduced that disk storage becomes practical for shared-prefix reuse, like prompt caching.

## 5. Empirical ablations worth showing on a slide

From §1 / Fig. 1 / §2.3.4:

1. **Fig. 1 right panel**: single-token FLOPs vs token position, and accumulated KV cache vs sequence length. Three curves (V3.2, V4-Pro, V4-Flash). Headline arrows: 3.7× / 9.8× FLOPs reduction; 9.5× / 13.7× KV reduction. This is the punchline slide visual.
2. **§2.3.4 last paragraph**: V4 KV cache ≈ 2% of BF16 GQA-8 baseline at 1M. Even more dramatic if compared to MHA.
3. **Hybrid scheme**: a slide diagramming the CSA-HCA-CSA-HCA-... layer interleaving, drawing the analogy to Gemma's local/global pattern but with two compression schemes.

## 6. Where it shipped + how production differs from paper

- **DeepSeek-V4-Pro** (1.6T total / 49B active): the flagship.
- **DeepSeek-V4-Flash** (284B total / 13B active): smaller, more aggressive compression ratios.
- Both shipped with **open weights** on HuggingFace at `deepseek-ai/DeepSeek-V4-Pro`; reference inference implementation is also open-sourced (footnote in §2.3 points to `inference/` in the HF repo).
- Trained on 32T tokens.
- Native 1M context, no separate context-extension finetune mentioned.

The paper IS the production paper (DeepSeek's house style — a "preview" model card published with full technical details). No expected delta paper → production.

---

## What V4 means for the talk's structure

V4 is the **canonical synthesis target** — it touches almost every category we cover:

| Category in our talk | V4 component | Notes |
|---|---|---|
| (a) KV cache compression | Token-level compressor in CSA & HCA; shared-KV MQA core | New axis: compression *along the sequence dim*, complementary to MLA's compression *along the per-head/latent dim* (which V3.2 used and V4 inherits). Worth contrasting on the (a) slide. |
| (a) KV storage | BF16/FP8 mixed-precision; FP4 indexer; on-disk KV | Reinforces the "FP8 KV cache" point in productized-without-paper list. |
| (c) Trainable sparse | Lightning indexer + top-$k$ in CSA | V4 paper explicitly credits DSA (§1: "CSA ... performs DeepSeek Sparse Attention (DSA)"). NSA not cited in V4, despite 8+ NSA authors appearing in V4 author list and architectural similarity. Likely-parallel research threads. |
| (c) Sliding window | Both CSA and HCA add small SWA branch | Same family as Mistral / Gemma local layers. |
| (c) Local/global interleaving | CSA ↔ HCA hybrid layers | Direct analog of Gemma local/global, Llama 4 iRoPE. |
| (c) Attention sinks | Learnable sink logits, cited to StreamingLLM | Bonus reuse. |
| Frontier numbers slide | 3.7× FLOPs, 9.5× KV vs V3.2; 50× KV vs BF16 GQA-8 baseline | Headline numbers for the "everything composes" punchline. |

**What V4 does NOT cover from our talk** (so we still need separate slides):
- The MQA → GQA → MLA progression (V4 uses MQA-core inside CSA/HCA, not GQA; MLA's latent compression is inherited from V3, not V4's contribution).
- Position encoding extension (V4 uses partial RoPE but doesn't innovate on context-extension methods like YaRN).
- FlashAttention kernels (V4 uses them but doesn't introduce kernel work).
- Ring Attention (V4 uses "contextual parallelism" per the TOC, not read yet).

## Things I did NOT verify in this conversation

- The exact CSA-to-HCA layer ratio in the interleaved scheme.
- The numerical values of $m$, $m'$, $k$, $n_\text{win}$, $c$, $d_c$, $n_h$, $n_h^I$ in V4's two models (Pro vs Flash) — would need §4.2.1 ("Model Setups") which I haven't read.
- The on-disk KV cache mechanism details — §3.5.2 not read.
- The training cost / wall clock numbers.
- Evaluation numbers beyond what Fig. 1 shows.

If we want any of these specifics on a slide, I should re-read the relevant sections.
