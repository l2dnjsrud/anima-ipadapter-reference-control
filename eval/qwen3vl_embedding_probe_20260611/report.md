# Qwen3-VL Embedding Probe

- Model: `Qwen/Qwen3-VL-Embedding-2B`
- Probe images: 6 identity128 color-panel references
- Output dimension: `2048`
- Instruction: represent manhwa/anime reference style, color palette,
  composition, character identity, and panel layout.
- Summary: `eval/qwen3vl_embedding_probe_20260611/summary.json`

Result:

- Off-diagonal cosine mean: `0.563116`
- Off-diagonal cosine min: `0.424949`
- Off-diagonal cosine max: `0.737467`

Interpretation: the model loads locally and produces separated image embeddings
for the reference set. The references do not collapse to one generic manhwa
vector, so this is a credible next encoder signal to test for adapter training.

Implementation note: the existing QwenVL plan assumed a `1024`-dim embedding.
The current public `Qwen/Qwen3-VL-Embedding-2B` model reports a default
`2048`-dim embedding with MRL/custom-dimension support. The native adapter path
should either use `2048` as the first QwenVL adapter input dimension or add an
explicit, documented projection/truncation stage before training.
