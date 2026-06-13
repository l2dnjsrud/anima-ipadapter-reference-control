# C103 post-C102/C098 다음 분기 결정

## 왜 C103을 열었나

C102는 local color 후보 64개를 Qwen3-VL-8B-Instruct로 다시 질의했지만
`confirmed_local_positive_count=0`이었다. 최소 학습 greenlight 기준은 `8`개였기 때문에,
이 후보 풀을 positive로 사용하면 녹색 소품, 배경, 말풍선, 일반 인간 얼굴을 잘못 학습할 위험이 크다.

동시에 C098은 C097 hard-shape 56 row로 deeper SigLIP encoder-LoRA를 학습했고 기능적으로는
성공했지만, 품질 승격에는 실패했다. `c098_lora_c094_w14` mean uplift는 `0.0865313863`으로
C096/C094와 비슷하거나 낮고, Qwen hard-shape baseline `0.1089544056`과는 차이가 크다.

따라서 C103은 새 학습을 바로 돌리는 단계가 아니라, 다음 루프가 같은 실패를 반복하지 않도록
분기를 결정하는 게이트다.

## 근거 요약

### local-real direct-green track

- C102 candidate rows: `64`
- covered rows: `64`
- confirmed local positive: `0`
- QA direct-green/non-human: `0`
- local negative: `63`
- unclear: `1`
- heldout leakage: `0`
- decision: `c103_blocked_needs_manual_annotation_or_external_teacher`

이 track은 사람이 C102 sheet에서 최소 `8`개 이상의 `local_positive`를 직접 확정하거나,
새 외부 teacher가 conflict-free positive를 만들어 주기 전까지 학습에 쓰지 않는다.

### external/synthetic hard-shape track

- C074 external real target positive: `10`
- C079 total target positives: `33`
- C080 paired direct-green rows: `196`
- C087 expanded crop-pair rows: `224`
- C097 selected hard-shape rows: `56`
- C097 explicit negatives: `56`
- heldout rows used: `0`

이 track은 local-real 증거는 아니지만, hard-shape/non-human reference-control을 계속 실험할 수 있는
현재 유일한 학습 후보군이다.

### SigLIP route 결과

- C092 Qwen-target distillation은 SigLIP 계열에서 큰 개선을 만들었다.
  - `c092_qwen_target_w14`: mean uplift `0.0852681653`
  - 하지만 Qwen baseline `0.1089544056`에는 못 미쳤고 green human face collapse가 남았다.
- C096 encoder-LoRA는 작동했지만 승격 실패.
  - `c096_lora_c094_w14`: mean uplift `0.0880849553`
- C098 deeper encoder-LoRA도 승격 실패.
  - `c098_lora_c094_w14`: mean uplift `0.0865313863`
  - heldout07 best C098 uplift: `0.0071462548`
  - visual audit: frog/chibi/mascot/non-human shape가 green adult humanoid face/bust로 수렴

## 버리는 분기

1. **C100-C102 local-real 후보 학습**
   - local positive가 `0`개라 아직 안전하지 않다.

2. **C098 같은 shallow/deeper SigLIP LoRA 반복**
   - C098은 loss와 loadability는 성공했지만 생성 품질은 C096/C094를 넘지 못했다.

3. **C074 external positive만으로 단독 학습**
   - target positive `10`개는 귀중하지만, license/NFA caveat와 캐릭터 domain 편향이 있어 단독 메인
     branch로 쓰기 어렵다.

## 선택한 C104 분기

선택 분기: `c104_expanded_qwen_target_siglip_distillation_gate`

핵심은 C092에서 효과가 가장 컸던 Qwen-target supervision을 C097의 더 큰 hard-shape 데이터로
확장하는 것이다. 단, 바로 긴 학습을 돌리지 않고 C104에서 먼저 manifest/probe를 만든다.

C104의 첫 목표:

- C097 56 row와 explicit negative를 유지한다.
- C087 Qwen hard-shape baseline target을 teacher signal로 연결할 수 있는지 확인한다.
- SigLIP feature margin이 C098의 완료된 LoRA signal보다 강한지 먼저 본다.
- probe가 막히면 학습하지 않고 blocker를 문서화한다.
- probe가 통과하면 bounded SigLIP adapter/encoder adaptation과 C098-style generation gate로 넘어간다.

## C104 통과/중단 조건

통과 전제:

- train rows `>=48`
- explicit negative rows == train rows
- heldout rows used == `0`
- missing path count == `0`
- probe margin이 C098 완료 LoRA signal보다 강하거나, 강하지 않다면 명확한 blocker를 낸다.

generation까지 갔을 때 최소 승격 후보 조건:

- mean uplift `>=0.100`
- heldout07 uplift `>=0.025`
- C104 blank-like rows `0`
- contact sheet에서 frog/chibi/mascot/non-human이 green adult humanoid face/bust로 무너지지 않을 것

## 다음 액션

C103은 `branch_decision.json` 기준으로 C104를 추가한다. C104는
`expanded Qwen-target SigLIP hard-shape distillation probe`로 시작하고, 완료 후에는 다시 문서화,
검증, 커밋, 푸시를 수행한다.
