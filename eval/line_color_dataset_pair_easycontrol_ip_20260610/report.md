# Line/Color Dataset Pair Colorization Probe

Date: 2026-06-10

## Inputs

- Line input: `/home/wktwin/anima-lora-training-bundle/image_dataset/MS-138/MS-138__MS_138-05_LLM.jpg`
- Color reference: `/home/wktwin/anima-lora-training-bundle/.pytest_cache/image_dataset_color/101-200/SG-138/SG-138-05.jpg`
- Output size: `1152x816`

## Attempts

1. PE IP-Adapter img2img only
   - Graph: line image -> VAEEncode -> KSampler, color image -> AnimaPEEncodeImage -> AnimaPEIPAdapterApply.
   - Result: color/style pressure was visible, but the original panel/page structure collapsed into newly composed figures.
   - Saved comparison: `/home/wktwin/anima-ipadapter-reference-control/eval/line_color_dataset_pair_20260610/comparison_sheet.jpg`

2. PE IP-Adapter img2img low-denoise
   - Graph: same as above, denoise lowered to `0.12/0.22/0.32`.
   - Result: still unstable. Lower denoise did not produce reliable source-preserving colorization.

3. AnimaEasyControl line control + PE IP-Adapter reference
   - Graph: line image -> AnimaEasyControlPatch using `anima_colorize_full.safetensors`, color image -> AnimaPEEncodeImage -> AnimaPEIPAdapterApply.
   - Settings:
     - `easy1p0_ip0p45`: EasyControl strength `1.0`, IP-Adapter strength `0.45`
     - `easy1p2_ip0p70`: EasyControl strength `1.2`, IP-Adapter strength `0.70`
   - Result: panel borders, character placement, and speech-bubble layout were preserved much better. Color was applied, but mostly as flat/gradient tinting rather than polished final dataset-quality coloring.

## Verdict

The current PE IP-Adapter is not enough by itself for dependable line-art colorization. It behaves as reference/style guidance, not spatial line control.

The practical path is to combine it with a line-control/colorize mechanism. The `AnimaEasyControlPatch + PE IP-Adapter` test is usable as a proof of direction, but it is not yet high-quality production colorization.

Best sample sheet:

- `/home/wktwin/anima-ipadapter-reference-control/eval/line_color_dataset_pair_easycontrol_ip_20260610/comparison_sheet.jpg`

