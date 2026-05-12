# c9 — DSA (DeepSeek Sparse Attention)

**Paper**: DeepSeek-AI, *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*, Dec 2025. [arXiv:2512.02556](https://arxiv.org/abs/2512.02556). [PDF](../papers/c9_DeepSeekV3.2_DSA.pdf).
**Length**: 23 pages. Read §1 (intro), §2 (DSA architecture + parity eval + inference costs) — pages 2-7 this session. Skipped §3 (Post-Training, GRPO scaling, RL).
**Why it matters**: **The first trainable-sparse-attention frontier model deployed at scale.** Production refinement of NSA (c8). Sets up V4's CSA. Shows that sparse-from-scratch works at frontier scale, not just at the 27B research scale NSA validated.

---

## 1. Bottleneck targeted

Same as NSA: $O(L^2)$ attention compute. DeepSeek-V3.2's stated goal (§1):

> "we first introduce DSA, a highly efficient attention mechanism designed to substantially reduce computational complexity. This architecture effectively addresses the efficiency bottleneck, preserving model performance even in long-context scenarios."

DSA is one of three contributions in V3.2 (alongside scaled GRPO and agentic data synthesis), but it's the **only architectural change** vs V3.1-Terminus. The rest of the architecture is identical.

## 2. Mechanism

§2.1 — DSA has two components.

### 2a. Lightning Indexer (eq 1)

A *cheap* query-side scorer that decides which preceding tokens are worth attending to. For query $h_t$ and a preceding token $h_s$:

$$I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}^I \cdot \text{ReLU}\bigl(q_{t,j}^I \cdot k_s^I\bigr)$$

Key choices that make this fast:
- $H^I$ is **small** (number of indexer heads, much less than the model's $H = 128$).
- **ReLU activation** instead of softmax — "for throughput consideration" (§2.1). No exp, no normalization across positions.
- Indexer query $q^I$, weight $w^I$ derived from the *latent* query vector ($c_t^Q$, MLA's compressed latent), partially RoPE'd.
- Indexer key $k_s^I$ derived from the preceding token's hidden state, partially RoPE'd.
- **Can be implemented in FP8** (per §2.1 and the V4 paper's earlier reference).

The indexer is still $O(L^2)$ as a kernel (you score every position), but with tiny constants — far cheaper than the main attention.

### 2b. Fine-grained Token Selection (eq 2)

Given index scores, select top-$k$ entries and run attention only on them:

$$u_t = \text{Attn}\bigl(h_t,\; \{c_s \mid I_{t,s} \in \text{Top-}k(I_{t,:})\}\bigr)$$

In V3.2 production: **$k = 2048$ tokens** per query (§2.1.1 Sparse Training Stage).

### 2c. Instantiated on top of MLA (§2.1 inset)

DSA is built *on top of* MLA. The key inheritance:

> "we implement DSA based on the MQA mode of MLA, where each latent vector (the key-value entry of MLA) will be shared across all query heads of the query token."

This is the same NSA insight: shared KV across heads is what makes the bandwidth wins of (a) compose with sparse selection. **One latent per cached token**, selected by the indexer, attended-to by all query heads.

Architecture diagram (Figure 2): input → down-project to latent $c^Q, c^{KV}$ → partial RoPE → indexer scores → top-$k$ selector → MQA core attention → output.

### How DSA differs from NSA

DSA is a **drastic simplification** of NSA's 3-branch design:

| | NSA (c8) | DSA (c9) |
|---|---|---|
| Compression branch | ✓ (coarse-grained) | ✗ (dropped) |
| Selection branch | ✓ (top-$n$ blocks) | ✓ (top-$k$ tokens via lightning indexer) |
| Sliding window | ✓ (separate) | ✗ (subsumed into top-$k$, which naturally picks nearby tokens) |
| Indexer | Compression scores reused (free) | **Dedicated lightning indexer** (small, ReLU, FP8) |
| Independent K, V per branch | ✓ | ✗ — only one branch |
| Top-$k$ unit | blocks ($n = 16$, block size $l' \approx 128$) | tokens ($k = 2048$) |

So DSA chose: one indexer + token-level top-$k$, with $k$ large enough (2048) that the selection naturally covers both local and long-range context. No need for separate branches.

## 3. Cost reduction — claimed and measured

### Asymptotic (§2.3)

Core attention: $O(L^2) \to O(Lk)$. Linear in context length once $k$ is fixed.

The indexer remains $O(L^2)$ but is much cheaper per operation (ReLU + FP8 + few heads).

### Measured wall-clock (Figure 3 — the punchline visual)

Cost per million tokens, on H800 clusters at $2/GPU-hour, as a function of token position:

| Phase | At 0 tokens | At 128K tokens |
|---|---|---|
| **Prefilling**: V3.1-Terminus | ~$0.07 | ~$0.7 |
| **Prefilling**: V3.2 (DSA) | ~$0.07 | **~$0.25** (~2.8× cheaper) |
| **Decoding**: V3.1-Terminus | ~$0.15 | ~$2.4 |
| **Decoding**: V3.2 (DSA) | ~$0.15 | **~$0.4** (~6× cheaper) |

Key shapes (verified by reading Figure 3):
- V3.1 curves grow roughly linearly with token position — attention dominates.
- V3.2 curves grow much more slowly — flat once you're past the indexer overhead.
- Decode improves more than prefill — consistent with attention being most bandwidth-bound during decode.

### Quality preservation (§2.2)

- **General benchmarks**: "no substantial performance degradation" vs V3.1-Terminus.
- **Human preference (ChatbotArena Elo)**: closely matched between V3.1 and V3.2.
- **AA-LCR long-context reasoning**: V3.2-Exp scores **4 points HIGHER** than V3.1 in reasoning mode.
- **Fiction.liveBench**: V3.2-Exp "consistently outperforms" V3.1-Terminus.

So like NSA: sparse-at-scale doesn't lose quality, and on long-context-specific evals actually wins.

## 4. Hardware-aware aspects

- **Lightning indexer in FP8** — extreme precision savings on the "small but $O(L^2)$" component.
- **ReLU instead of softmax** — eliminates the exp + normalization (more parallel, better Tensor Core utilization).
- **MQA-mode MLA** — one latent per token, shared across all heads. Inherits NSA's GQA-compatibility insight.
- **Open-sourced inference implementation** — paper links to `https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference`.
- **Short-context fallback**: §2.3 mentions a "masked MHA mode to simulate DSA" for short sequences — DSA's overhead would dominate if $L$ is small, so they use a masked-dense kernel below some threshold.

## 5. Training procedure (§2.1.1) — non-trivial

DSA can't be trained naively from scratch — the top-$k$ selection has to be aligned with what a *dense* model would do, otherwise the indexer's gradients are noisy.

### Stage 1: Dense Warm-up (eq 3)
- **Keep dense attention** active; **freeze all params except the lightning indexer**.
- Indexer trained with **KL-divergence to the dense attention's distribution**:
  $$\mathcal{L}^I = \sum_t \mathbb{D}_{\text{KL}}\bigl(p_{t,:} \;\|\; \text{Softmax}(I_{t,:})\bigr)$$
  where $p_{t,:}$ is the L1-normalized sum of attention scores across all heads in the main model.
- **2.1B tokens** total (1000 steps × 16 sequences × 128K tokens), LR = $10^{-3}$.

This is essentially **knowledge distillation from the dense attention into the indexer**. The indexer learns to predict which tokens dense attention actually attends to.

### Stage 2: Sparse Training (eq 4)
- Switch on top-$k$ selection. Train **all params end-to-end**.
- Indexer still aligns to dense, but only on the selected token set:
  $$\mathcal{L}^I = \sum_t \mathbb{D}_{\text{KL}}\bigl(p_{t,S_t} \;\|\; \text{Softmax}(I_{t,S_t})\bigr)$$
- **Detach the indexer input from the computational graph** — main model trains via language modeling loss; indexer trains separately via $\mathcal{L}^I$. No gradient interference.
- **943.7B tokens** (15000 steps × 480 sequences × 128K tokens), LR = $7.3 \times 10^{-6}$.

### Why this matters

This is a clever staged approach: dense warm-up gives the indexer a sane starting target (otherwise it'd be selecting random positions during sparse training). The detach trick is the same gradient-isolation logic NSA used for "branch-independent K, V" — but here it's between the indexer and the main model, not between branches.

## 6. Where it shipped + how production differs from research

### Shipped
- **DeepSeek-V3.2-Exp** (Sept 2025): the experimental release with DSA.
- **DeepSeek-V3.2** (Dec 2025): final production version of V3.2, same DSA, refined post-training (RL via scaled GRPO).
- **DeepSeek-V3.2-Speciale**: reasoning-focused variant.
- Open weights on HuggingFace; open-sourced inference implementation.

V3.2 is **continued pre-trained from V3.1-Terminus** — they didn't retrain from scratch. Starting checkpoint has 128K context already extended. So the cost numbers compare V3.2 (DSA) vs V3.1-Terminus (dense MLA + extended context).

### V3.2 vs NSA (research → production transitions)

| | NSA | DSA |
|---|---|---|
| Scale validated at | 27B params, 270B tokens | V3 full scale (~685B), 943.7B continued-pre-training tokens |
| Top-$k$ unit | Blocks of $\sim 128$ tokens, $n = 16$ | Tokens, $k = 2048$ |
| Selection mechanism | Reused compression-branch scores | Dedicated lightning indexer |
| Training | From scratch | Continued pre-training from dense checkpoint |
| Branches | 3 (cmp + slc + win) | 1 (just selection) |
| Shipped frontier model | No | Yes (V3.2 family) |

### V4 lineage

V4 then re-adds the compression branch and sliding window — its **CSA** is essentially DSA's "lightning indexer + selection" path plus NSA's compression branch plus a sliding window. So V4 = NSA's 3-branch design but with DSA's lightning indexer doing the selection. V4 also adds **HCA** (heavily-compressed attention) layers interleaved with CSA layers.

The lineage (per V4 paper's explicit claim): **DSA → V4 (with compression on top)**. V4 §1: *"CSA compresses the KV caches along the sequence dimension and then performs DeepSeek Sparse Attention (DSA)"*. V4 paper does NOT cite NSA — despite 8+ shared authors and architectural similarity (V4 CSA's 3-component design = compression + selection + sliding ≈ NSA's 3 branches). Our reading: NSA and DSA were probably *parallel* research threads at DeepSeek-AI, with NSA's ideas informing V4 implicitly through shared authors. The papers don't say this; we're inferring from author overlap + architectural similarity + training-strategy alignment (NSA and V4 both from scratch; DSA continued PT).

---

## Talk-relevant takeaways

- **DSA is the first frontier-deployed trainable sparse attention.** This is the "shipped" punchline for the (c) section — NSA was research, DSA shipped, V4 evolved it further.
- **The architectural simplification is the surprise.** NSA had 3 branches with branch-independent K,V; DSA has 1 branch with one big top-$k$. Production validated that you don't need explicit branch separation if $k$ is large enough.
- **The lightning indexer is the load-bearing component.** Small heads, ReLU not softmax, FP8 — all hardware-aware choices to make the "still $O(L^2)$ but cheap" indexer fast.
- **2-stage training (dense warm-up + sparse) is non-trivial.** Without the warm-up, the indexer has no target. The KL-distillation-from-dense framing is a clean way to bootstrap.
- **Concrete cost wins**: at 128K context, **6× cheaper decode** and **3× cheaper prefill** vs the same model with dense attention. On H800 GPUs.
- **Quality is preserved or improved**: same on short context, better on long-context reasoning (AA-LCR, Fiction.liveBench).

## Things I did NOT verify in this session

- §3 (Post-Training, scaling GRPO, RL setup) — not read this session. Probably not directly relevant to the talk's attention focus.
- §4+ (Evaluation Results, Real-World Tasks) — only saw summary claims in §1 and §2.2.
- **The exact V3.2 param count** — paper §2 says "exactly the same architecture as DeepSeek-V3.2-Exp" and continued-trained from V3.1-Terminus, which is V3.1 (685B total / 37B active per token, per V3 paper). Did NOT see explicit V3.2 param count in pages I read; should verify before putting on a slide.
- The "indexer can be FP8" claim — paper text mentions FP8 in §2.1, but I'd want to confirm against §3 or inference implementation if precise.

## Slide implications

For the (c) section closing, DSA gets 1-2 slides. Suggested structure:

**Slide 1 — "DSA — production trainable sparse attention"**:
- Figure 2 (the DSA architecture with lightning indexer feeding the top-$k$ selector)
- Three bullets:
  - Lightning indexer scores every preceding token (small heads, ReLU, FP8 — still $O(L^2)$ but cheap)
  - Top-$k$ selector: pick the best $k = 2048$ tokens
  - MQA-core attention on selected tokens — inherits MLA's KV compression
- Simplification vs NSA: 1 branch instead of 3. Production validated this.

**Slide 2 (optional) — "DSA shipped: real wall-clock costs"**:
- Figure 3 (cost per million tokens vs context length)
- 6× decode / 3× prefill at 128K
- Quality matches V3.1 on short context, beats it on long-context reasoning
- First trainable-sparse-attention frontier model.

Could be combined into one slide if we're aggressive.
