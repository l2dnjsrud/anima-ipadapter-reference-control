# C105 stronger encoder / teacher route decision

## 목적

C104에서 C097 hard-shape 데이터의 SigLIP2 token feature 분리를 다시 확인했지만, 가장 좋은
`mean_max_token`도 margin `0.01997534079211105`, AUC `0.7228954081632653`에 그쳤다. 이는
Qwen hard-shape baseline `0.1089544056`과 AUC gate `0.85`를 모두 넘지 못한다.

따라서 C105는 긴 학습을 바로 시작하지 않고, C106에서 실제로 돌릴 다음 branch 하나를 고르는
route gate로 진행했다.

## 확인한 근거

| 근거 | 핵심 수치 | 판단 |
|---|---:|---|
| C102 local-real QA | confirmed local positives `0` / minimum `8` | 로컬 실사 후보 직접 학습 불가 |
| C087 Qwen baseline | mean uplift `0.10895440559772807` | 현재 hard-shape 최강 기준 |
| C098 SigLIP encoder-LoRA | best mean uplift `0.08653138633881173` | 학습/생성은 됐지만 Qwen baseline 아래 |
| C104 SigLIP token probe | best margin `0.01997534079211105`, AUC `0.7228954081632653` | SigLIP-only 추가 학습 신호 부족 |
| C064 hard failure embedding | QwenVL `1/3`, SigLIP `0/3`, PE `1/3` | 기존 embedding만으로는 hard failure 전체 해결 불가 |

## 선택한 route

선택 branch는 `qwen_teacher_distillation`이다.

이유:

- C087 QwenVL 쪽이 C098 generation gate에서 여전히 가장 강한 hard-shape baseline이다.
- C102에서 local-real positive가 0개라서 로컬 후보를 직접 positive로 쓰는 것은 안전하지 않다.
- C104가 SigLIP-only feature signal 부족을 보여줬으므로, C106은 Qwen/VL teacher target 또는
  teacher feature를 사용해 student projection/adapter를 먼저 검증해야 한다.

## C106 시작 조건

C106은 바로 긴 학습이 아니라 다음 순서로 진행한다.

1. C097 train rows와 explicit negative를 유지한 Qwen-teacher distillation manifest를 만든다.
2. Qwen/VL teacher feature 공간과 student projection 공간에서 positive/negative 분리 가능성을 먼저 probe한다.
3. probe가 통과할 때만 bounded training을 진행한다.
4. training이 진행되면 C098-style hard-shape generation gate로 `no_ip`, C087 Qwen baseline, C098 best,
   C106 candidate를 비교한다.

## C106 pass/stop gate

pre-training gate:

- positive rows >= `48`
- explicit negative rows == positive rows
- heldout rows used == `0`
- missing path count == `0`
- teacher feature margin >= `0.05`
- teacher feature AUC >= `0.85`
- C104 best margin `0.01997534079211105` 초과

generation promotion gate:

- mean uplift >= `0.1089544056`
- heldout07 uplift >= `0.025`
- improved rate >= `0.91`
- blank-like rows == `0`
- visual audit에서 non-human silhouette / side-profile / frog-chibi body collapse가 개선될 것

즉 C106은 단순히 loss가 줄었다고 통과하지 않는다. C087 Qwen baseline 이상으로 생성 품질이 올라가야
reference-control 후보로 승격한다.

## 기각한 route

`stronger_siglip_checkpoint_adaptation`은 지금 바로 C106 primary route로 선택하지 않는다. C098은 이미
성공적으로 학습/생성됐지만 Qwen baseline 아래였고, C104도 SigLIP feature 분리 신호가 약했다.

`manual_or_external_annotation`은 fallback으로 유지한다. C102의 0-positive 문제를 직접 해결할 수 있지만,
새 annotation 또는 외부 데이터 권리 검토가 필요해서 즉시 자동 루프로 돌릴 branch는 아니다.

`local_real_direct_green_training_from_c102`는 C102 confirmed local positive가 0개라서 기각한다.

## 산출물

- `eval/c105_stronger_encoder_teacher_route_gate_20260613/source_inventory.json`
- `eval/c105_stronger_encoder_teacher_route_gate_20260613/route_decision.json`
- `docs/c105_stronger_encoder_teacher_route_decision_ko.md`

## 다음 단계

C106은 `qwen_teacher_distillation` branch로 시작한다. 첫 산출물은
`docs/c106_qwen_teacher_feature_distillation_plan_ko.md`와
`eval/c106_qwen_teacher_feature_distillation_20260613/manifest_summary.json` /
`probe_summary.json`이다.
