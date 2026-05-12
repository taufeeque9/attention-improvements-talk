# A Decade of Improving *Attention*

Slides and supporting material for a talk on post-2017 attention improvements that make long-context agentic LLMs possible.

- **Slides** — live deck: <https://taufeeque9.github.io/attention-improvements-talk/> (all animations preserved). Source in [slides/](slides/) ([slides.md](slides/slides.md)); see [slides/README.md](slides/README.md) for `pnpm dev` / build / PDF export.
- **Notes** — per-paper reading notes in [notes/](notes/).
- **Papers** — local PDFs in [papers/](papers/), prefixed by talk section (`a*` KV compression, `c*` sparse attention, `e*` alternatives / hybrid SSMs).
- **Reference** — [reference.md](reference.md): every concrete number on the slides traced to a derivation.

## Reading list

### (a) KV-cache compression

| Paper | PDF | Notes |
|---|---|---|
| MQA — Shazeer 2019 | [pdf](papers/a1_MQA_Shazeer2019.pdf) | [notes](notes/a1_MQA.md) |
| GQA — Ainslie 2023 | [pdf](papers/a2_GQA_Ainslie2023.pdf) | [notes](notes/a2_GQA.md) |
| MLA — DeepSeek-V2 | [pdf](papers/a3_MLA_DeepSeekV2.pdf) | [notes](notes/a3_MLA.md) |
| DeepSeek-V3 | [pdf](papers/a3b_DeepSeekV3.pdf) | [notes](notes/a3b_DeepSeekV3.md) |
| PagedAttention (vLLM) | [pdf](papers/a4_PagedAttention_vLLM.pdf) | — |
| Sarathi-Serve (chunked prefill) | [pdf](papers/a5_SarathiServe_ChunkedPrefill.pdf) | — |

### (c) Sparse attention

| Paper | PDF | Notes |
|---|---|---|
| Sparse Transformer — Child 2019 | [pdf](papers/c1_SparseTransformer_Child2019.pdf) | [notes](notes/c1_SparseTransformer.md) |
| Longformer (SWA) | [pdf](papers/c2_Longformer_SWA.pdf) | [notes](notes/c2_Longformer.md) |
| Mistral 7B | [pdf](papers/c3_Mistral7B.pdf) | — |
| Gemma 2 / Gemma 3 | [g2](papers/c4_Gemma2.pdf) · [g3](papers/c5_Gemma3.pdf) | — |
| Command A (Cohere) | [pdf](papers/c6_CommandA_Cohere.pdf) | — |
| StreamingLLM (attention sinks) | [pdf](papers/c7_StreamingLLM_AttentionSinks.pdf) | — |
| NSA — DeepSeek | [pdf](papers/c8_NSA_DeepSeek.pdf) | [notes](notes/c8_NSA.md) |
| DSA — DeepSeek-V3.2 | [pdf](papers/c9_DeepSeekV3.2_DSA.pdf) | [notes](notes/c9_DSA.md) |
| MoBA — Moonshot | [pdf](papers/c10_MoBA_Moonshot.pdf) | — |
| Ring Attention | [pdf](papers/c11_RingAttention.pdf) | — |
| Gemini 1.5 | [pdf](papers/c12_Gemini1.5.pdf) | — |
| DeepSeek-V4 | [pdf](papers/c13_DeepSeekV4.pdf) | [notes](notes/c13_DeepSeekV4.md) |

### (e) Alternatives / hybrids

| Paper | PDF |
|---|---|
| Mamba — Gu & Dao 2023 | [pdf](papers/e1_Mamba_GuDao2023.pdf) |
| Mamba 2 (SSD) | [pdf](papers/e2_Mamba2_SSD.pdf) |
| Jamba / Jamba 1.5 (AI21) | [v1](papers/e3_Jamba_AI21.pdf) · [v1.5](papers/e4_Jamba1.5_AI21.pdf) |
| Nemotron-3 Super (NVIDIA) | [pdf](papers/e5_Nemotron3Super_NVIDIA.pdf) |

### Foundation

| Paper | PDF |
|---|---|
| Attention Is All You Need — Vaswani 2017 | [pdf](papers/AttentionIsAllYouNeed_Vaswani2017.pdf) |
