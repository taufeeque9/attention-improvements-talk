# a3 — Multi-Head Latent Attention (MLA)

**Paper**: DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, June 2024. [arXiv:2405.04434](https://arxiv.org/abs/2405.04434). [PDF](../papers/a3_MLA_DeepSeekV2.pdf).
**Length**: 52 pages. Read for MLA: §2.1 (pp. 6–9), §C (full formulas, p. 31), §D ablations (pp. 31–32).
**Why it matters**: MLA is the **first non-incremental change** to KV-cache compression since MQA/GQA. Different topology: not "share fewer heads" but "compress per-token KV into a low-rank latent and absorb the up-projection into Q/O at inference."

---

## 1. Bottleneck targeted

**Same as MQA/GQA**: decoder inference is bandwidth-bound by the KV cache. From §2.1:

> "during generation, its heavy Key-Value (KV) cache will become the bottleneck that limits the inference efficiency. … MQA and GQA … require a smaller magnitude of KV cache, but their performance does not match MHA."

MLA's pitch: get **smaller-than-GQA** KV cache **and** **better-than-MHA** quality, simultaneously.

## 2. Mechanism — three pieces

### 2a. Low-rank KV joint compression (§2.1.2, eq 9–11)

Replace per-head K, V with a single compressed latent. For input $h_t$:

$$c_t^{KV} = W^{DKV} h_t, \quad c_t^{KV} \in \mathbb{R}^{d_c}, \;\; d_c \ll d_h n_h$$

This is what gets cached. K and V are recovered (at training time only) via up-projection:

$$k_t^C = W^{UK} c_t^{KV}, \qquad v_t^C = W^{UV} c_t^{KV}$$

For DeepSeek-V2: $d_c = 4 d_h = 512$. Compare to MHA's per-token K+V size of $2 n_h d_h = 2 \cdot 128 \cdot 128 = 32{,}768$ elements. **Compression factor ~64×** in raw element count.

### 2b. Absorption at inference (the deep trick)

At inference, $W^{UK}$ and $W^{UV}$ are **never materialized** to compute K, V. Instead:

- $W^{UK}$ is **absorbed into $W^Q$**: precompute $\tilde{W}^Q = (W^{UK})^\top W^Q$ once. Per query: $\tilde q_t = \tilde W^Q h_t$. Attention score is $\tilde q_t^\top c_j^{KV}$ — a $d_c$-dim dot product against the cached latent.
- $W^{UV}$ is **absorbed into $W^O$**: the value-weighted-sum produces a $d_c$-dim output per head; then the absorbed $W^O \tilde W^{UV}$ projects to $d$.

Per cached token, we load **only $d_c$ floats** (≈ the latent), not $2 n_h d_h$ K+V floats. And we never expand them.

This is what makes MLA practical. Without absorption, MLA would still need to re-materialize K, V at every decode step from the cached latent (so you'd save bandwidth on the cache but pay back FLOPs on decompression).

### 2c. Decoupled RoPE (§2.1.3, eq 14–19) — necessary engineering twist

**The problem**: RoPE rotates keys by a position-dependent matrix $R_t$. If applied to $k_t^C = W^{UK} c_t^{KV}$, then $k_t^C \to R_t W^{UK} c_t^{KV}$ — the $R_t$ gets between $W^{UK}$ and the cached latent. Now you can't absorb $W^{UK}$ into $W^Q$ because matrix multiplication isn't commutative through $R_t$.

> "we must recompute the keys for all the prefix tokens during inference, which will significantly hinder the inference efficiency."

**The fix**: split K and Q into two parallel parts:
- $c$-part (compressed, no RoPE): $k_{t,i}^C, q_{t,i}^C$ from the latent, dim $d_c$ — absorbable.
- $R$-part (rotary, RoPE'd): per-head queries $q_t^R \in \mathbb{R}^{d_h^R}$ and one **shared** key $k_t^R \in \mathbb{R}^{d_h^R}$ that carries position. NOT absorbable, but small.

Per-head concatenation: $q_{t,i} = [q_{t,i}^C; q_t^R]$, $k_{t,i} = [k_{t,i}^C; k_t^R]$.

For DeepSeek-V2: $d_h^R = d_h / 2 = 64$.

**Total cached per token**: $(d_c + d_h^R) \cdot l$ elements (where $l$ = number of layers). The latent + the shared RoPE key.

## 3. Cost reduction — claimed and measured

### KV cache size

From Table 1 (§2.1.4) — KV cache per token across mechanisms:

| Method | KV cache per token | DeepSeek-V2 setting |
|---|---|---|
| MHA | $2 n_h d_h l$ | $2 \cdot 128 \cdot 128 \cdot l = 32{,}768\, l$ |
| GQA-$G$ | $2 G d_h l$ | $G$ tunable, e.g. $G{=}8 \Rightarrow 2048\, l$ |
| MQA | $2 d_h l$ | $256\, l$ |
| **MLA** | $(d_c + d_h^R)\, l$ | $(512 + 64)\, l = 576\, l$ |

MLA at $d_c = 4 d_h, d_h^R = d_h/2$ gives KV cache equivalent to **GQA-2.25** — between GQA-2 and GQA-3 — but with MHA-or-better quality.

### Empirical ablations (Appendix D)

**Table 8 — 7B dense, MHA vs GQA-8 vs MQA**, 1.33T tokens, param-matched:

| Benchmark | MQA (7.1B) | GQA-8 (6.9B) | **MHA (6.9B)** |
|---|---|---|---|
| BBH | 33.2 | 35.6 | **37.0** |
| MMLU | 37.9 | 41.2 | **45.2** |
| C-Eval | 30.0 | 37.7 | **42.9** |
| CMMLU | 34.6 | 38.4 | **43.5** |

→ At dense 7B scale on hard benchmarks, **GQA-8 leaves quality on the table** (~4 MMLU points behind MHA). MQA loses another ~4 points. This is the strongest empirical evidence we have that the GQA Pareto picture is incomplete: on hard benchmarks, MHA's full KV head count matters.

**Table 9 — MoE models, MHA vs MLA**, at two scales:

| | Small MoE MHA | Small MoE MLA | Large MoE MHA | Large MoE MLA |
|---|---|---|---|---|
| Activated Params | 2.5B | 2.4B | 25.0B | 21.5B |
| Total Params | 15.8B | 15.7B | 250.8B | 247.4B |
| **KV / token (elements)** | **110.6K** | **15.6K** | **860.2K** | **34.6K** |
| KV ratio (MLA / MHA) | — | **14%** | — | **4%** |
| BBH | 37.9 | **39.0** | 46.6 | **50.7** |
| MMLU | 48.7 | **50.0** | 57.5 | **59.0** |
| C-Eval | **51.6** | 50.9 | 57.9 | **59.2** |
| CMMLU | 52.3 | **53.4** | 60.7 | **62.5** |

→ At large MoE scale: **MLA's KV is 4% of MHA's** (25× smaller), and MLA WINS on 4/4 benchmarks (BBH +4.1, MMLU +1.5, C-Eval +1.3, CMMLU +1.8). MLA is not a quality compromise — it's a quality improvement.

### Headline DeepSeek-V2 numbers (abstract)

vs DeepSeek-67B (previous dense MHA model):
- **42.5% training cost reduction** (mostly from MoE, not MLA)
- **93.3% KV cache reduction** (this is the MLA contribution)
- **5.76× max generation throughput**

## 4. Hardware-aware aspects

The absorption trick is the heart of the hardware-aware design. **By never materializing per-head K, V**, MLA avoids the apparent paradox that low-rank compression "saves bytes but costs FLOPs to decompress." The decompression matrices fold into Q, O at compile time; inference loads only the latent.

### Arithmetic intensity (per the talk's framing)

Per cached token at inference, with batch $b$, all $H$ heads, ignoring the small $d_h^R$ part:
- **Bytes loaded**: $d_c$ floats × 2 (fp16) = $2 d_c$ bytes per token (one latent — shared across all H query heads)
- **FLOPs**: $H \cdot 2 \cdot d_c$ for QK^T scores + $H \cdot 2 \cdot d_c$ for AV → $4 H d_c$
- **Arithmetic intensity**: $4 H d_c / 2 d_c = 2H$ FLOPs/byte

For DeepSeek-V2 ($H = 128$): **MLA arithmetic intensity ≈ 256 FLOPs/byte** — comparable to MQA at the same $H$, and within striking distance of the H100 peak ~290.

Compare:
- MHA: 1 FLOP/byte
- GQA-8 ($H{=}32$): 4 FLOPs/byte
- MQA ($H{=}32$): 32 FLOPs/byte
- **MLA ($H{=}128$): ~256 FLOPs/byte** — nearly compute-bound

The key difference vs MQA: MLA has high $H$ AND the cache is a *latent* the query heads can address differently via their absorbed $W^{UK}$ projections. MQA at high $H$ would have all queries reading literally the same K, V; MLA has them reading the same latent but extracting different projections of it. **MQA-like bandwidth, MHA-like (or better) expressivity.**

## 5. Empirical ablations worth showing on a slide

1. **Figure 3** — the 4-panel illustration MHA / GQA / MQA / MLA. The MLA panel shows the H queries fanning out to a single shared "Compressed Latent KV" rectangle. Pedagogically perfect for showing topological difference vs GQA's "$G$ groups."
2. **Table 1** — KV-cache-per-token across methods. Shows MLA $\approx$ GQA-2.25 in cache size.
3. **Table 9** — the MLA vs MHA head-to-head: 4% KV cache, better on 4/4 benchmarks. Headline punchline.
4. The arithmetic-intensity computation: MLA = ~256 FLOPs/byte for DeepSeek-V2 — a great connect-back to our per-step framework.

## 6. Where it shipped + how production differs from paper

**Shipped in**: DeepSeek-V2, V2.5, V3, V3.2, R1, R1-0528, V4 (V4 inherits MLA per its paper, then adds CSA/HCA *on top*). Also adopted in **Kimi K2** (Moonshot 2025, per V4 paper's references and Kimi tech report).

**Open weights**: yes (HuggingFace). DeepSeek-V2 was the first frontier model to ship MLA. As of mid-2025, MLA is the **DeepSeek family's signature attention** but has NOT been adopted by Llama, Mistral, Qwen, Gemma, or other open-frontier lineages — they all stuck with GQA. (Hypotheses for why: training stability concerns, simpler infrastructure, GQA's H/G knob is easier to retrofit.)

**Production realities not in paper**:
- The absorbed-projection trick requires the inference framework to know about it. Naive implementations of MLA inference miss the absorption and run slower than MHA.
- V3.2 and V4 ADD sparse attention on top of MLA (DSA, CSA) — MLA is a baseline, not a destination.

---

## Talk-relevant takeaways

- **MLA is a topology change, not a knob change.** GQA's H/G is on a Pareto curve between MHA and MQA. MLA leaves the curve: latent compression + absorption is structurally different from head-sharing.
- **The absorption trick is the deep magic.** Without it, MLA is just "compress KV with extra cost." With it, MLA achieves the arithmetic intensity of MQA while retaining the per-head expressivity of MHA.
- **Decoupled RoPE is the engineering scar** showing that the absorption trick almost didn't work. RoPE breaks absorbability; the fix is to carry position info on a small separate channel.
- **MLA at $H = 128$ hits ~256 FLOPs/byte** — for the first time, decode is *nearly* compute-bound on H100. This is the strongest single illustration in the talk of "decode is bandwidth-bound" being a *solvable* problem.
- **Empirical evidence is real**: Table 9 shows MLA dominates MHA in both KV size AND quality. Not a trade-off.
- **Not universally adopted**: only DeepSeek family + Kimi. Worth raising as a slide-aside.

## Things I did NOT verify in this session

- That Kimi K2 actually uses MLA. From the V4 paper I read earlier I noted Kimi K2 is listed among MLA adopters in plan.md, but I haven't read the Kimi tech report. Should verify when we get to that paper (not on our list yet).
- That production frameworks (vLLM, TensorRT-LLM) implement the absorption trick correctly. The paper says it's possible; whether production has it is implementation-dependent.
- The "no Llama/Mistral/Qwen adopted MLA" claim. Negative evidence is harder to verify than positive evidence — I'm inferring from the public tech reports for those families (which all describe GQA, not MLA). Will sanity-check when we read those.
- The arithmetic-intensity calculation (~256 FLOPs/byte) is my derivation in this session, following the same pattern I used for MHA and GQA. Cross-checked: MLA bytes = $d_c + d_h^R$ per token (loaded once across all H queries); FLOPs = $H \cdot 2 \cdot (d_c + d_h^R) \cdot 2$ for QK^T + AV. Ratio ≈ $2H$. Confirmed.

## Slide implications

For the (a) section of the talk, MLA should get its own slide (currently a placeholder). Suggested structure:
1. **Figure 3** (4-panel MHA/GQA/MQA/MLA) + one-sentence "different topology"
2. The math: latent compression + absorption trick + decoupled RoPE — three steps
3. Numbers: Table 9 MLA-vs-MHA punchline + arithmetic-intensity computation showing ~256 FLOPs/byte

If we want it shorter: a single 2-slide block (one for the structural change with Figure 3, one for the numbers + arithmetic intensity).
