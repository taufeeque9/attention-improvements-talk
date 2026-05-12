# a3b — DeepSeek-V3 (attention-focused notes)

**Paper**: DeepSeek-AI, *DeepSeek-V3 Technical Report*, Dec 2024. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437). [PDF](../papers/a3b_DeepSeekV3.pdf).
**Length**: 53 pages. Read attention-relevant sections this session: §1 intro, §2.1 architecture (MLA section), §3.2.3 memory savings, §3.3 FP8 (precision-with-attention parts), §3.4 inference deployment, §4.1 data, §4.2 hyperparameters, §4.3 long-context extension. **Skipped**: §3.2 DualPipe details, §3.3 FP8 details for MoE GEMMs, §4.4-§4.5 evaluations, §5 post-training (SFT, RL), appendices.
**Why it matters for the talk**: V3 is what V3.1-Terminus (and downstream V3.2/DSA) was built on top of. Tells us *how MLA actually trains at frontier scale* — what tricks make it work, what context length, what precision choices.

---

## 1. Architecture — same MLA as V2, with specific V3-scale hyperparameters

§2.1.1: "DeepSeek-V3 adopts the MLA architecture." The equations (1-11) are *identical* to V2's MLA. No architectural changes vs V2.

**V3 model dimensions** (§4.2):
- Layers: **61**
- Hidden dimension: **$d = 7168$**
- MLA heads: **$n_h = 128$** (much more than the V2's 128 — wait, V2 also had 128, just confirming the V3.2 number we used earlier)
- Per-head dim: **$d_h = 128$**
- **KV compression dim: $d_c = 512$** (so 4× per-head, matches the "GQA-2.25-equivalent" claim from V2 paper)
- **Query compression dim: $d_c' = 1536$**
- **Decoupled RoPE per-head dim: $d_h^R = 64$** (half of $d_h$, matches V2)
- Total params: **671B**; activated per token: **37B**
- MoE: 1 shared + 256 routed experts, top-8 active, max 4 nodes per token routing

So the MLA "256 FLOPs/byte" arithmetic-intensity number we put on the MLA slide was computed with **exactly these V3 hyperparameters** ($H=128, d_c=512$). Verified.

## 2. Training: short context base + long-context extension

§4.2 Training Hyper-Parameters:

- **Base pretraining sequence length: 4K tokens.** Trained on 14.8T tokens at this length.
- AdamW optimizer ($\beta_1=0.9, \beta_2=0.95$, weight decay 0.1)
- LR schedule: linear warmup → constant $2.2 \times 10^{-4}$ for first 10T tokens → cosine decay to $2.2 \times 10^{-5}$ over 4.3T tokens → constants for the final 500B
- Batch size: ramped from 3072 → 15360 over first 469B tokens, then constant
- Gradient clipping norm: 1.0
- 2048 H800 GPUs, HAI-LLM framework

**This confirms the speculation from prior turns: V3 pretrained at 4K context.** All the heavy $O(n^2)$ attention math during pretraining was at modest $n = 4096$, where the absolute cost is tractable even without sparse attention.

### Long-context extension (§4.3) — YaRN, two-phase

After base pretraining:

1. **Phase 1: extend 4K → 32K**
   - 1000 steps
   - Sequence length 32K, batch 1920
   - LR $7.3 \times 10^{-6}$ (matches final pretraining LR)
2. **Phase 2: extend 32K → 128K**
   - 1000 steps
   - Sequence length 128K, batch 480
   - Same LR

Both phases use **YaRN** ([Peng et al. 2023a](https://arxiv.org/abs/2309.00071)), applied *exclusively to the decoupled shared key $k_t^R$*. YaRN config (identical to V2):
- $s = 40$
- $\alpha = 1$, $\beta = 32$
- Scaling factor $\sqrt{t} = 0.1 \ln s + 1$

§4.3 closing: needle-in-a-haystack at 128K passes (Figure 8).

Cost (Table 1 / §1): context extension is **119K GPU-hours** = ~$0.238M, vs 2664K GPU-hours = ~$5.328M for pretraining. **Context extension is ~4.5% of pretraining cost.**

## 3. Attention-specific training optimizations

The actually-novel-for-attention parts of V3's training framework:

### 3a. MLA up-projection recompute (§3.2.3)

> "Recomputation of RMSNorm and MLA Up-Projection. We recompute all RMSNorm operations and MLA up-projections during back-propagation, thereby eliminating the need to persistently store their output activations."

This is specifically MLA-aware activation checkpointing:
- MLA's $W^{UK}, W^{UV}$ up-projections produce full $[H, n, d_h]$ K, V tensors during the forward pass.
- Storing those activations for backprop is expensive.
- V3 throws them away after forward, recomputes during backward.
- Cost: extra FLOPs during backward (one extra matmul for the up-projection).
- Benefit: large activation-memory savings.

This is the answer to the user's earlier question about "does MLA help training memory?" — **yes**, via the recompute trick. The latent $c_t^{KV}$ is what's stored; full per-head K, V is recomputed when needed.

### 3b. Attention stays in BF16; everything else goes FP8 (§3.3.1)

V3's FP8 mixed-precision framework runs *most* compute in FP8 (the linear matmuls — $Fprop$, $Dgrad$, $Wgrad$). But:

> "we maintain the original precision (e.g., BF16 or FP32) for the following components: the embedding module, the output head, MoE gating modules, **normalization operators, and attention operators**."

So **attention is NOT in FP8** at training. They explicitly kept it at BF16 because the attention operator is precision-sensitive (softmax overflow risk + the small magnitudes of attention scores).

### 3c. Custom E5M6 FP8 format for activations feeding into attention's backward (§3.3.3)

> "**Inputs of the Linear after the attention operator.** These activations are also used in the backward pass of the attention operator, which makes it sensitive to precision. **We adopt a customized E5M6 data format exclusively for these activations.**"

So even though the activations are FP8, V3 uses an **E5M6 variant** (5-bit exponent, 6-bit mantissa) instead of the standard E4M3 (4-exp, 3-mant) or E5M2 (5-exp, 2-mant). The extra mantissa bits trade dynamic range for precision — important because attention's gradients are sensitive.

Additionally: "these activations will be converted from an `1x128` quantization tile to an `128x1` tile in the backward pass. To avoid introducing extra quantization error, all the scaling factors are round scaled (integral power of 2)."

So: attention sits at the intersection of "needs higher precision than FP8" and "feeds activations into things that *do* run in FP8." V3 handles this with a special tile-tile rotation + integral scaling.

## 4. Inference deployment (§3.4) — attention does use TP

Worth noting because it contradicts the "no TP" framing of training:

**Prefill (§3.4.1)**: deployed across 4 nodes / 32 GPUs minimum. **Attention runs with TP4 + Sequence Parallelism** + 8-way DP. MoE runs with EP32.

**Decode (§3.4.2)**: deployed across 40 nodes / 320 GPUs minimum. **Attention runs with TP4 + SP** + 80-way DP. MoE runs with EP320.

> "Unlike prefilling, **attention consumes a larger portion of time in the decoding stage**. Therefore, we overlap the attention of one micro-batch with the dispatch + MoE + combine of another."

Confirms: at deployment, attention is the bottleneck at decode (consistent with our talk's "decode is bandwidth-bound" framing). The MQA-mode of MLA + KV cache sharing handle the bandwidth side. V3.2's DSA on top adds the sparse selection.

## 5. Data packing: no cross-sample attention mask (§4.1, footnote-style mention)

> "we implement the document packing method for data integrity but **do not incorporate cross-sample attention masking** during training."

Multiple documents are packed into a single 4K training sequence without an attention mask between them. Saves complexity at the cost of letting documents attend to each other (presumably the model just learns to mostly ignore irrelevant prior context, given the 14.8T training scale).

## 6. Training cost summary (Table 1, §1)

| Phase | H800 GPU-hours | Cost @ $2/GPU-hr |
|---|---|---|
| Pre-training | 2,664K | $5.328M |
| Context extension | 119K | $0.238M |
| Post-training (SFT + RL) | 5K | $0.01M |
| **Total** | **2,788K** | **$5.576M** |

Pre-training is 95.5% of cost. Context extension is 4.3%. Post-training is negligible (0.2%).

The headline cost number (~$5.5M total for a frontier-class MoE model) made waves when V3 dropped — comparable closed models were rumored to cost an order of magnitude more.

## Talk-relevant takeaways

- **V3 confirms MLA's contribution is the *whole* "decode bandwidth" half of the talk's framing.** Architecture inherited from V2, hyperparameters ($d_c = 512, H = 128, d_h = 128$) are the basis for our "MLA ≈ 240 FLOPs/byte" calculation. No new attention architecture in V3.
- **V3 pretrained at only 4K context** — that's how a fully dense $O(n^2)$ attention was tractable for 14.8T tokens. Then YaRN-extended to 32K, then 128K, each phase only 1000 steps and just 4.3% of total cost.
- **V3 does use MLA-specific training tricks**: up-projection recompute (saves activation memory; trades a backward matmul for cache savings), BF16-attention-everything-else-FP8, custom E5M6 for the linear-after-attention.
- **V3 does NOT use any train-time sparse attention.** Dense MLA throughout. The talk's "V3 only attacks the decode bandwidth wall" framing is verified.
- The **context-extension recipe (YaRN, two-phase 32K→128K)** is the same as V2's. Worth a mention if we ever do a position-encoding-extension slide.
- Worth noting: **attention DOES use TP at inference** (TP4) even though training is TP-free. This is because per-GPU memory budget differs.

## Things I did NOT verify

- §3.2 DualPipe details (pipeline parallelism). Not attention-relevant.
- §3.3 FP8 GEMM specifics (fine-grained quantization, accumulation precision, mantissa-over-exponents). MoE-relevant, not attention.
- §3.5 Hardware suggestions. Out of scope.
- §4.4 Evaluation results. Not architectural.
- §4.5 Ablation studies (MTP, load balancing). Not attention.
- §5 Post-training. Not attention.
- Appendices.

## Slide implications

The user asked for a slide on V3 training. Suggested:

**Slide: "V3 — how dense MLA actually trained at frontier scale"** (one slide):

- Hyperparameters: 671B / 37B-active MoE, 61 layers, $H = 128$, $d_h = 128$, $d_c = 512$. 14.8T tokens.
- Pretrained at **4K context** (where $O(n^2)$ is tractable). 2.66M GPU-hours.
- Long-context extension: **two-phase YaRN** (4K → 32K → 128K), 1000 steps each, **4.3% of total cost**.
- Three MLA-specific training tricks:
  1. **MLA up-projection recompute** during backprop (trades a backward matmul for big activation-memory savings)
  2. **Attention stays in BF16** while linear layers go FP8 (precision-sensitive)
  3. **Custom E5M6 FP8** for the linear-after-attention activations (extra mantissa bits)
- Total cost: $5.5M — frontier-class at ~1/10th the rumored cost of closed competitors.

One slide should be enough — V3's attention contributions are real but modest beyond "MLA + extension recipe." More slides would mostly be DualPipe / FP8 / MoE infrastructure which is out of scope.
