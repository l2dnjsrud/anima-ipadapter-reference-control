# Identity Feature Probe

- Encoder: `Qwen/Qwen3-VL-Embedding-2B`
- Manifest: `eval/c106_qwen_teacher_feature_distillation_20260613/probe_manifest.jsonl`
- Pairs: `112`
- Positive mean: `0.8937567493745259`
- Negative mean: `0.6760281973651477`
- Separation margin: `0.21772855200937813`
- Pairwise AUC: `1.0`
- Midpoint accuracy: `1.0`
- Decision: `feature_separates_proxy_pairs`

## Anchor Group Summaries

| group | positive mean | negative mean | margin | AUC | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| c082_frog_yokai_guard | 0.8950152210891247 | 0.6837664134800434 | 0.21124880760908127 | 1.0 | feature_separates_proxy_pairs |
| c082_goblin_mage | 0.9088799133896828 | 0.6674169525504112 | 0.24146296083927155 | 1.0 | feature_separates_proxy_pairs |
| c082_green_oni_scout | 0.8857029303908348 | 0.6864117085933685 | 0.19929122179746628 | 1.0 | feature_separates_proxy_pairs |
| c082_jade_lizard_monk | 0.8771011158823967 | 0.6570072323083878 | 0.22009388357400894 | 1.0 | feature_separates_proxy_pairs |

Positive and negative labels are read from the manifest.
This gate checks feature separation only; it does not prove generation quality.
