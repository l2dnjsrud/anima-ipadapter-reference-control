# C108 Qwen Teacher Embedding Alignment Generation Gate

- Decision: `c108_generation_gate_not_promoted`
- Samples: `40`
- Variants: `3`
- Results: `120`
- PNG count: `120`
- Blank count: `0`
- Contact sheet train: `eval/c108_qwen_teacher_embedding_alignment_gate_20260615/contact_sheet_train.jpg`
- Contact sheet heldout: `eval/c108_qwen_teacher_embedding_alignment_gate_20260615/contact_sheet_heldout.jpg`
- Cleanup: `eval/c108_qwen_teacher_embedding_alignment_gate_20260615/cleanup_port_8116.txt`

## PE Metric

| variant | split | cases | mean cosine | mean uplift | improved rate |
|---|---|---:|---:|---:|---:|
| `blend_species_face` | `train` | `32` | `0.8177628647536039` | `0.06273302435874939` | `0.8125` |
| `blend_species_face` | `heldout` | `8` | `0.8040817752480507` | `0.05353397876024246` | `0.875` |
| `blend_species_face` | `all` | `40` | `0.8150266468524933` | `0.060893215239048004` | `0.825` |
| `c108_qwen_teacher_alignment_w14` | `train` | `32` | `0.804932227358222` | `0.04990238696336746` | `0.71875` |
| `c108_qwen_teacher_alignment_w14` | `heldout` | `8` | `0.7464450895786285` | `-0.0041027069091796875` | `0.625` |
| `c108_qwen_teacher_alignment_w14` | `all` | `40` | `0.7932347998023033` | `0.03910136818885803` | `0.7` |

- PE C108 vs blend all: `{'cases': 40, 'left_wins': 14, 'ties': 0, 'mean_left_minus_right': -0.021791847050189973}`

## QwenVL Metric

| variant | split | cases | mean cosine | mean uplift | improved rate |
|---|---|---:|---:|---:|---:|
| `blend_species_face` | `train` | `32` | `0.8204112350940704` | `0.04612010158598423` | `0.8125` |
| `blend_species_face` | `heldout` | `8` | `0.8346929922699928` | `0.02647087723016739` | `0.75` |
| `blend_species_face` | `all` | `40` | `0.8232675865292549` | `0.042190256714820865` | `0.8` |
| `c108_qwen_teacher_alignment_w14` | `train` | `32` | `0.8113647699356079` | `0.037073636427521706` | `0.75` |
| `c108_qwen_teacher_alignment_w14` | `heldout` | `8` | `0.8210786208510399` | `0.012856505811214447` | `0.75` |
| `c108_qwen_teacher_alignment_w14` | `all` | `40` | `0.8133075401186943` | `0.03223021030426025` | `0.75` |

- QwenVL C108 vs blend all: `{'cases': 40, 'left_wins': 13, 'ties': 0, 'mean_left_minus_right': -0.009960046410560608}`

## 해석

C108은 `no_ip` 대비 reference-control이 켜진다. 그러나 PE/QwenVL 양쪽에서 current best `blend_species_face`를 넘지 못했고, contact sheet 육안 감사에서도 확실한 개선이 없다. 따라서 C108 standalone checkpoint는 승격하지 않는다.

다음 루프는 target embedding alignment를 더 오래 반복하는 것보다, current best blend와 조합 가능한 routing 또는 비인간 실루엣/말풍선/크롭 context를 직접 잡는 데이터/손실 쪽으로 넘어가야 한다.
