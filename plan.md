# Attention Improvements Talk — 45 min

## Goal

Audience already knows the 2017 *Attention is All You Need* transformer. Give them a **deep** understanding of the post-2017 attention-module improvements that actually made long-context agentic workloads possible at the frontier — not a shallow tour.

## Structure (working draft)

- **(5 min) Framing.** Recap original multi-head attention module. Show two cost walls for agentic workloads at scale:
  - Prefill: O(N²) compute per layer.
  - Decode: O(N) KV-memory bandwidth per generated token; KV cache grows linearly with context.
  - Why the vanilla MHA breaks at 100k–1M-token agent contexts.
- **(13 min) Category (a) — KV-cache compression.** MQA → GQA → MLA. Attacks the decode-bandwidth wall.
- **(13 min) Category (c) — Sparse / long-context attention patterns.** SWA → trainable sparse (NSA → DSA, MoBA). Attacks the prefill-compute wall.
- **(5 min) Category (e) — The parallel thread: SSM / linear hybrids.** Replacing attention rather than improving it. Mamba → Mamba-2 → Jamba → Nemotron 3 Super. Frame: same bottleneck (O(n²) compute, O(n) KV), completely different answer (linear-time SSM recurrence in most layers, attention only where recall demands it).
- **(3–5 min, time permitting) Category (d) — kernels.** FlashAttention v1→v3 + PagedAttention as the systems glue. Likely a brief mention rather than a deep dive.
- **(remaining) Synthesis + open questions.** Show how a current frontier model — **DeepSeek V4** (Apr 2026) — composes (a) + (c). Then contrast with **Nemotron 3 Super** (Mar 2026) as the same long-context-agentic goal reached via a fundamentally different architecture. Open question to the audience: do these two threads converge, or stay separate?

---

## Papers — Category (a): KV-cache / inference efficiency

All downloaded to [papers/](papers/). Annotation = frontier model(s) that adopted the technique, with source.

| # | Paper | arXiv | File | Frontier adoption |
|---|---|---|---|---|
| a1 | **Multi-Query Attention (MQA)** — Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, 2019 | [1911.02150](https://arxiv.org/abs/1911.02150) | [a1_MQA_Shazeer2019.pdf](papers/a1_MQA_Shazeer2019.pdf) | **PaLM** (Chowdhery et al. 2022), **Falcon** (TII), **StarCoder**. Largely superseded by GQA. |
| a2 | **Grouped-Query Attention (GQA)** — Ainslie et al., 2023 | [2305.13245](https://arxiv.org/abs/2305.13245) | [a2_GQA_Ainslie2023.pdf](papers/a2_GQA_Ainslie2023.pdf) | **De-facto standard** for open frontier: **Llama 2/3/3.1/4** ([Llama 3 herd](https://arxiv.org/abs/2407.21783)), **Mistral 7B / Mixtral**, **Qwen 2.5/3**, **Gemma 2/3**, **Phi-3/4**, **Command A**. |
| a3 | **Multi-Head Latent Attention (MLA)** — DeepSeek-V2, 2024 | [2405.04434](https://arxiv.org/abs/2405.04434) | [a3_MLA_DeepSeekV2.pdf](papers/a3_MLA_DeepSeekV2.pdf) | **DeepSeek V2 / V3 / V3.2 / R1**, **Kimi K2** (Moonshot). |
| a3b | **DeepSeek-V3 technical report** (production use of MLA at scale) | [2412.19437](https://arxiv.org/abs/2412.19437) | [a3b_DeepSeekV3.pdf](papers/a3b_DeepSeekV3.pdf) | DeepSeek V3 / R1. Companion reading for MLA — see how it interacts with MTP and FP8 training. |
| a4 | **PagedAttention / vLLM** — Kwon et al., SOSP 2023 | [2309.06180](https://arxiv.org/abs/2309.06180) | [a4_PagedAttention_vLLM.pdf](papers/a4_PagedAttention_vLLM.pdf) | Serving substrate for most open-weight frontier deployment. Influences TensorRT-LLM, SGLang block-manager design. |
| a5 | **Sarathi-Serve** — Agrawal et al., 2024 (chunked prefill + stall-free batching) | [2403.02310](https://arxiv.org/abs/2403.02310) | [a5_SarathiServe_ChunkedPrefill.pdf](papers/a5_SarathiServe_ChunkedPrefill.pdf) | Chunked prefill is the default in vLLM / SGLang / TensorRT-LLM long-context serving. |

### Productized (no canonical paper) — mention in slides

- **Prefix / prompt caching** — Anthropic ([blog](https://www.anthropic.com/news/prompt-caching)), OpenAI automatic prompt caching, Google Gemini context caching, DeepSeek API context caching. Load-bearing for agentic workloads.
- **FP8 / INT8 KV-cache quantization** — production default on H100/H200 in TensorRT-LLM and vLLM. Specific deployment numbers not disclosed by frontier labs but is the default config in serving stacks targeting those models.

### Optional anchor numbers for the framing slide — to add if time allows

From [Reiner Pope podcast cross-check](resources/reiner_pope_podcast.md) (validated subagent report):

- **HBM-drain invariant**: HBM capacity / HBM bandwidth ≈ **25 ms** on modern GPUs (A100 = 40 ms, H100 = 24 ms, B200 = 23 ms, H200 = 29 ms). Roughly constant because capacity and bandwidth grow together. Hard floor on decode latency when memory is full.
- **Empirical ~2 KB/token KV pressure** from reverse-engineering Gemini's 200K context API pricing.
- **3–5× input/output API price gap** = direct evidence current frontier deployments are bandwidth-bound on decode.
- **Context lengths stuck at 100K–200K for 2+ years** = empirical proof the memory wall is biting now.

Reiner direct quote candidates (verify before slide use):
- "Prefill is compute-limited and decode is memory bandwidth-limited."
- "On most GPUs, this ends up being somewhere around 300" (FLOPs/byte ratio).
- "[Attention] is mostly dominated by memory fetches rather than matrix multiplies."
- "Sparse attention gives you a get-out, because you get this square root."

---

## Papers — Category (e): SSM / linear-time hybrids (parallel thread)

A separate research direction reaching the same goal — long-context, agentic-tractable models — by **replacing** attention with state-space recurrences in most layers, and keeping only a few attention layers for precise associative recall. Same bottleneck framing applies: $O(n^2)$ compute and $O(n)$ KV bandwidth are the things to escape; SSMs escape them by being linear-time in $n$ and constant-state in $b$.

| # | Paper | arXiv | File | Frontier adoption |
|---|---|---|---|---|
| e1 | **Mamba** — Gu & Dao, 2023 (selective SSM with hardware-aware scan) | [2312.00752](https://arxiv.org/abs/2312.00752) | [e1_Mamba_GuDao2023.pdf](papers/e1_Mamba_GuDao2023.pdf) | Foundational. Selective state spaces with a parallel scan kernel. |
| e2 | **Mamba-2 / SSD** — Dao & Gu, 2024 (state-space duality with linear attention) | [2405.21060](https://arxiv.org/abs/2405.21060) | [e2_Mamba2_SSD.pdf](papers/e2_Mamba2_SSD.pdf) | The bridge that connects SSMs to linear attention — duality showed they're the same family. Used in Mamba-2 hybrid stacks. |
| e3 | **Jamba** — Lieber et al., AI21, Mar 2024 (first frontier hybrid Mamba + attention + MoE) | [2403.19887](https://arxiv.org/abs/2403.19887) | [e3_Jamba_AI21.pdf](papers/e3_Jamba_AI21.pdf) | First production hybrid LM. Interleaves Mamba layers with attention layers + MoE. |
| e4 | **Jamba 1.5** — AI21, Aug 2024 (production-scaled Jamba family) | [2408.12570](https://arxiv.org/abs/2408.12570) | [e4_Jamba1.5_AI21.pdf](papers/e4_Jamba1.5_AI21.pdf) | **AI21 Jamba 1.5 Mini / Large** (398B total / 94B active). Validates the hybrid recipe at frontier scale. |
| e5 | **Nemotron 3 Super** — NVIDIA, Apr 2026 (current frontier hybrid Mamba-Attention MoE for agentic reasoning) | [NVIDIA Tech Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf) | [e5_Nemotron3Super_NVIDIA.pdf](papers/e5_Nemotron3Super_NVIDIA.pdf) | **NVIDIA Nemotron 3 Super** (120B total / 12B active MoE, 1M context). NVFP4 pretraining, LatentMoE, MTP. Open weights on HF. Explicitly billed for "agentic reasoning." |

---

## Papers — Category (c): Sparse / long-context attention patterns

| # | Paper | arXiv | File | Frontier adoption |
|---|---|---|---|---|
| c1 | **Sparse Transformer** — Child, Gray, Radford, Sutskever, 2019 (strided + fixed factorized attention) | [1904.10509](https://arxiv.org/abs/1904.10509) | [c1_SparseTransformer_Child2019.pdf](papers/c1_SparseTransformer_Child2019.pdf) | **GPT-3** — uses "alternating dense and locally banded sparse attention" per [GPT-3 paper §2.1](https://arxiv.org/abs/2005.14165). |
| c2 | **Longformer** — Beltagy, Peters, Cohan, 2020 (sliding window + global tokens) | [2004.05150](https://arxiv.org/abs/2004.05150) | [c2_Longformer_SWA.pdf](papers/c2_Longformer_SWA.pdf) | Originator of SWA pattern used in Mistral 7B and later Gemma 2/3 local layers. |
| c3 | **Mistral 7B** — Jiang et al., 2023 (productionized SWA at frontier-7B scale) | [2310.06825](https://arxiv.org/abs/2310.06825) | [c3_Mistral7B.pdf](papers/c3_Mistral7B.pdf) | **Mistral 7B / Mixtral 8x7B / Mistral-style models**. |
| c4 | **Gemma 2** — Google DeepMind, 2024 (interleaved local 4k SWA + global 8k, 1:1 ratio) | [2408.00118](https://arxiv.org/abs/2408.00118) | [c4_Gemma2.pdf](papers/c4_Gemma2.pdf) | **Gemma 2** (2B / 9B / 27B). |
| c5 | **Gemma 3** — Google DeepMind, 2025 (5:1 local-to-global, 1024-token local windows, longer context) | [2503.19786](https://arxiv.org/abs/2503.19786) | [c5_Gemma3.pdf](papers/c5_Gemma3.pdf) | **Gemma 3** family. |
| c6 | **Command A / Command R7B / Aya Expanse** — Cohere, 2025 (3:1 SWA-RoPE + Full-NoPE interleaving) | [2504.00698](https://arxiv.org/abs/2504.00698) | [c6_CommandA_Cohere.pdf](papers/c6_CommandA_Cohere.pdf) | **Cohere Command A**. Same family of pattern as Llama 4 iRoPE — worth comparing. |
| c7 | **StreamingLLM / Attention Sinks** — Xiao, Tian, Chen, Han, Lewis, 2023 | [2309.17453](https://arxiv.org/abs/2309.17453) | [c7_StreamingLLM_AttentionSinks.pdf](papers/c7_StreamingLLM_AttentionSinks.pdf) | Inference-time technique shipped in HuggingFace Transformers, TensorRT-LLM. Borderline for "frontier" — bolted on, not baked into pretraining. |
| c8 | **Native Sparse Attention (NSA)** — Yuan et al., DeepSeek, Feb 2025 (trainable sparse: token compression + block selection) | [2502.11089](https://arxiv.org/abs/2502.11089) | [c8_NSA_DeepSeek.pdf](papers/c8_NSA_DeepSeek.pdf) | DeepSeek's research line leading to DSA. |
| c9 | **DeepSeek-V3.2** introducing **DSA (DeepSeek Sparse Attention)** — Dec 2025 | [2512.02556](https://arxiv.org/abs/2512.02556) | [c9_DeepSeekV3.2_DSA.pdf](papers/c9_DeepSeekV3.2_DSA.pdf) | **DeepSeek V3.2 / V3.2-Speciale**. Lightning indexer + fine-grained token selection on top of MLA; O(L²)→O(Lk). Confirmed via verified PDF abstract. |
| c10 | **MoBA (Mixture of Block Attention)** — Moonshot AI, Feb 2025 | [2502.13189](https://arxiv.org/abs/2502.13189) | [c10_MoBA_Moonshot.pdf](papers/c10_MoBA_Moonshot.pdf) | **Kimi** long-context serving (per paper §1). MoE-style routing over attention blocks. |
| c11 | **Ring Attention with Blockwise Transformers** — Liu, Zaharia, Abbeel, 2023 | [2310.01889](https://arxiv.org/abs/2310.01889) | [c11_RingAttention.pdf](papers/c11_RingAttention.pdf) | **Large World Model** (Liu et al. 2024). **Rumored** for Gemini 1.5's 1M context — cited as concurrent work in c12, not confirmed. |
| c12 | **Gemini 1.5 Pro** — Google, 2024 (1M+ context frontier model; cites Ring Attention as concurrent work) | [2403.05530](https://arxiv.org/abs/2403.05530) | [c12_Gemini1.5.pdf](papers/c12_Gemini1.5.pdf) | Companion / context reading. Closed model — use only for "what frontier scale looks like" framing, not architecture claims. |
| c13 | **DeepSeek-V4** — DeepSeek-AI, Apr 2026 (hybrid attention: CSA + HCA; 1M-token native context) | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | [c13_DeepSeekV4.pdf](papers/c13_DeepSeekV4.pdf) | **DeepSeek-V4-Pro** (1.6T total / 49B active MoE) and **DeepSeek-V4-Flash** (284B / 13B active). vs V3.2 at 1M context: 27% FLOPs / 10% KV (Pro); 10% FLOPs / 7% KV (Flash). Verified by reading paper §2.3, §2.3.4. **Primary synthesis target.** |

---

## Reading order (suggested)

Order optimized for the talk narrative, not chronology.

1. **a1 MQA** → **a2 GQA** → **a3 MLA** → **a3b DeepSeek-V3** (KV-compression arc; the cleanest story in the talk)
2. **a4 PagedAttention** → **a5 Sarathi-Serve** (systems for keeping the compressed cache fed)
3. **c1 Sparse Transformer** → **c2 Longformer** → **c3 Mistral 7B** → **c4 Gemma 2** → **c5 Gemma 3** → **c6 Command A** (static sparsity arc; sets up the "why we need *learned* sparsity")
4. **c7 StreamingLLM** (bridging idea — attention sinks)
5. **c8 NSA** → **c9 DSA (DeepSeek-V3.2)** → **c10 MoBA** → **c13 V4 hybrid CSA + HCA** (trainable sparse arc + the latest compression-on-top-of-sparse step)
7. **e1 Mamba** → **e2 Mamba-2 (SSD)** → **e3 Jamba** → **e4 Jamba 1.5** → **e5 Nemotron 3 Super** (SSM / linear-hybrid parallel thread — read AFTER the attention papers, since the contrast is what makes this section interesting)
6. **c11 Ring Attention** + **c12 Gemini 1.5** (orthogonal: distributed long-context for training/inference; closed-model context)

---

## Per-paper content to extract (do for each)

For each paper read, capture in a per-paper notes file:

1. **The bottleneck it targets** (in one sentence — prefill compute? decode bandwidth? KV memory? context length?)
2. **The exact mechanism** (the equation or block diagram that changed vs. baseline MHA)
3. **The cost reduction claim** (asymptotic + measured), and what it costs in quality
4. **What's hardware-aware about it** (if anything — e.g. MLA's absorption-into-Q, NSA's GQA-aligned block selection)
5. **Empirical ablations worth showing** — one or two figures from the paper to put on a slide
6. **Where it shipped** and how the production version differs from the paper

---

## Empirical compare-runs (user's idea — to scope later)

User mentioned wanting to run these models directly to compare what improved. Options to scope:

- **Like-for-like attention-only swaps** on a small open base (e.g. swap MHA → GQA → MLA on Llama-3 8B-ish or Qwen 0.5B): inference latency, KV cache size, throughput at varying context length, quality on a long-context eval (RULER, LongBench, needle-in-haystack).
- **Frontier-as-is comparisons**: serve DeepSeek-V3.2 (MLA+DSA) vs Llama 3.1 70B (GQA, dense) vs Mistral-NeMo (GQA+SWA) on the same long-context workload. Easier to actually run; comparison is across many confounders, not just attention.

We can decide once paper-reading is further along — open question whether the talk needs original measurements or whether the paper-reported numbers suffice.

---

## Things I have NOT independently verified yet (verify when reading)

- All adoption claims for **Qwen3** (QK-Norm, RoPE θ=1M, GQA) — need to check the Qwen3 report directly.
- **Gemma 2 ratio "1:1"** and **Gemma 3 ratio "5:1, 1024-local"** — verify against c4 and c5 directly.
- **MLA "~93% KV reduction"** — verify against c3 paper directly.
- **MoBA used in production Kimi serving** — verify against c10 §1.
- **GPT-3's exact sparse pattern** — verify against c1 vs GPT-3 paper §2.1.

The adoption column above came from the research-agent's web search. The actual cited *papers* are downloaded and trustworthy; the *adoption tags* should be cross-checked against each paper's introduction.
