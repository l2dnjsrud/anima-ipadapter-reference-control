# c078 visual audit

## 판정

`ready_for_c079_training_manifest`

c078은 text-only Anima/Qwen generation으로 synthetic direct-green/non-human reference 후보 24장을 생성했다. 생성은 모두 성공했고 blank image는 없었다.

## 라벨 요약

- `target_positive`: 23
- `reject_unclear`: 1
- 신규 `target_positive`: 23

## 시각 판단

대부분의 후보는 green skin이 얼굴과 몸에 명확히 나타나고, pointed ears, horns, tail, scales, wings, antennae, plant/monster silhouette 같은 non-human cue가 있다. c077 public sample source와 달리 green hair/outfit만 있는 false-positive가 아니라, 학습 목표인 direct-green/non-human reference trait에 직접 맞는다.

`c078_synth_21`은 두 캐릭터가 함께 생성되어 single-character 학습 기준에 맞지 않으므로 `reject_unclear`로 제외했다.

## 결론

c078 synthetic bootstrap source는 c079 학습 manifest의 신규 target-positive source로 사용할 수 있다. 단, synthetic source만으로 최종 품질을 보장하지는 않으므로 c079에서는 c074/c078 positives와 guard/proxy negatives를 섞고, 결과는 반드시 기존 clean32+heldout8 및 direct-green focus gate에서 다시 검증해야 한다.
