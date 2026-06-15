# c078 synthetic direct-green bootstrap plan

## 목적

c077은 public/HF sample source를 넓혔지만 신규 direct-green/non-human `target_positive`가 0개였다. 그래서 c078은 더 학습하기 전에 **synthetic direct-green reference 후보를 직접 만들면 학습 전제로 쓸 수 있는지** 확인한다.

## 생성 방식

- 서버: `/data/ai/comfyui02`, API `http://127.0.0.1:8102`
- 모델: `anima-base-v1.0.safetensors`
- text encoder: `qwen_3_06b_base.safetensors`
- VAE: `qwen/qwen_image_vae.safetensors`
- 생성 방식: IP-Adapter 없이 text-only Anima/Qwen generation
- 후보 수: 24 prompts
- 해상도: 768 x 1024
- sampler: `er_sde`, steps `18`

## 프롬프트 원칙

- 반드시 `single character`를 넣는다.
- 반드시 `green skin`, `non-human`, species cue를 넣는다.
- goblin, lizardfolk, oni, slime, alien, frog yokai, dragonkin, orc, serpent folk, plant monster 등 다양한 형태를 만든다.
- negative에는 multiple characters, normal human skin, text/watermark, nude/nsfw를 넣는다.

## 승격 조건

다음 c079 학습 manifest로 넘어가려면 다음 조건을 만족해야 한다.

- 생성 성공 이미지 24장 이상
- blank 0
- 신규 visually confirmed `target_positive` 12장 이상
- heldout rows used 0
- raw generated images는 `.tmp/c078_synthetic_direct_green_bootstrap/` 아래에만 두고 커밋하지 않는다.

## 결과 요약

c078은 24장 생성에 성공했고 blank는 0개다. contact sheet 수동 검수 결과 23장이 `target_positive`, 1장은 multiple characters 때문에 `reject_unclear`다. 따라서 c078 decision은 `ready_for_c079_training_manifest`다.
