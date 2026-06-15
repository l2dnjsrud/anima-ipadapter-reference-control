# C108 Qwen Teacher Embedding Alignment Training Gate

- Decision: `proceed_to_c108_generation_gate`
- Failures: `[]`
- Manifest rows: `56`
- Heldout rows used: `0`
- Missing path count: `0`
- Steps: `128`
- Rows loaded: `56`
- Explicit negative rows: `56`
- First loss: `0.6143436431884766`
- Final loss: `0.5449631214141846`
- Mean loss: `0.6113223081920296`
- Mean teacher loss: `1.003852569963783`
- Finite loss: `True`
- Trainable parameters: `528384`
- Frozen base parameters: `4947838963`
- Init checkpoint: `checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors`
- Output checkpoint: `checkpoints/anima_qwenvl_ip_adapter_c108_teacher_alignment_b128_0128_20260615.safetensors`
- Checkpoint loadable: `True`
- PE checkpoint rejected: `True`

## 해석

C108 학습은 QwenVL teacher embedding target alignment loss를 실제 학습 루프에 넣고 56개 hard-negative row에서 128 step bounded training을 수행했다. Loss는 finite이고 checkpoint는 loadable이며 PE checkpoint rejection guard도 통과했으므로 C001 기준에서는 generation gate로 넘어갈 수 있다.

다만 이 결과는 아직 생성 품질 승격을 뜻하지 않는다. 다음 C002에서 isolated ComfyUI/API로 `no_ip`, current best `blend_species_face`, `c108_qwen_teacher_alignment_w14`를 같은 reference set에서 비교해야 한다.
