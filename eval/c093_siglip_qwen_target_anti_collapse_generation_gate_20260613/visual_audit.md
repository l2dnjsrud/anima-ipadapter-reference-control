# c093 SigLIP Qwen-Target Anti-Collapse Visual Audit

## 결론

`c093_anti_collapse_w14`는 수치상 C092 w14를 아주 조금 넘었지만, 실제 contact sheet 기준으로는 promotion하지 않는다.

- decision: `c093_anti_collapse_not_promoted`
- best c093 variant: `c093_anti_collapse_w14`
- c093 w14 mean uplift: `0.08637357797831002`
- c092 w14 mean uplift: `0.08526816531384505`
- Qwen baseline mean uplift: `0.10895440559772807`
- heldout07 c093 w14 uplift: `0.004518958388039396`
- heldout07 c092 w14 uplift: `0.0009986629992987384`

평균 uplift는 C092보다 `+0.00110541266446497` 높지만, contact sheet에서 보이는 개선은 거의 없다. frog/chibi/mascot/non-human row가 여전히 green human head 또는 bald face template으로 빨려 들어간다. 따라서 이 결과는 "loss와 shape proxy가 약간 오른 trainability signal"이지, "reference-control 품질 개선"으로 보기 어렵다.

## 관찰

- `crop_pair00`부터 `crop_pair09`까지 C093 w10/w12/w14는 대체로 C092 w10/w14와 같은 방향이다.
- 일부 row에서 색감과 얼굴 음영은 안정되지만, reference 고유의 몸 비율, chibi silhouette, frog mascot 형태는 유지하지 못한다.
- C093의 anti-collapse 학습은 c092 collapse 이미지를 explicit negative로 사용했지만, 출력 분포 자체는 여전히 사람형 녹색 얼굴로 수렴한다.
- `heldout07`은 학습에서 제외한 검증 row인데, monster side-profile, red eye, exaggerated jaw 형태가 유지되지 않고 human warrior profile로 이동한다.
- pixel audit의 blank-like 1장은 `crop_pair00_no_ip`라 C093 checkpoint 자체의 blank 실패는 아니다.

## 판단

C093는 "작동한다"와 "좋다"를 분리해서 보면 전자만 만족한다.

- ComfyUI loader selector에 checkpoint가 표시된다.
- API workflow로 110장 생성이 끝났다.
- checkpoint는 loadable이고 PE checkpoint reject guard도 통과했다.
- 하지만 reference shape fidelity는 C092 대비 실질 개선이 없다.

따라서 다음 루프는 단순 contrastive continuation이 아니라, encoder-side adaptation 또는 target/image-space loss를 더 직접적으로 걸어야 한다. 특히 SigLIP embedding에서 non-human/chibi/frog 형태가 human face attractor로 붕괴하는 문제를 막으려면, explicit negative만으로는 부족하다.

## 다음 결정

다음 실험은 C094로 분리한다.

- C092 checkpoint를 best SigLIP base로 유지한다.
- C093 checkpoint는 promotion하지 않고 regression/evidence artifact로 보존한다.
- 다음 루프 후보:
  - Qwen baseline target을 유지하면서 reference crop의 silhouette/edge/segmentation target을 추가한다.
  - SigLIP image encoder 출력에 작은 adapter/LoRA를 붙여 non-human shape cluster를 더 직접적으로 학습한다.
  - single-character/non-human focused manifest를 별도로 구성하고, frog/chibi/mascot row를 과표집한다.
  - heldout07 같은 side-profile monster를 검증 전용으로 계속 유지한다.
