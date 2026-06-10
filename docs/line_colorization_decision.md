# Line-Art Colorization Decision

Date: 2026-06-10

## Decision

Stop pursuing PE IP-Adapter-only line-art colorization.

Continue with a separate SigLIP2/TimeResampler/IPCrossAttn training stage for
high-quality Anima reference-control. Treat line-art colorization as a separate
spatial-control problem that needs EasyControl/ControlNet-like conditioning in
addition to any reference adapter.

## Evidence

The PE IP-Adapter checkpoint is functional as a reference-control adapter:

- ComfyUI full contact-sheet evaluation passed.
- Best scale: `1.0`
- Mean PE uplift over no-IP: `+0.0937`
- Improved rate: `87.5%`
- Evidence: `eval/comfy_pe_full_contactsheet_20260610/report.md`

However, line-art colorization failed when PE IP-Adapter was used as the only
control source:

- The reference color/style affected generation.
- The original panel/page structure collapsed into newly composed figures.
- Lower denoise settings did not reliably preserve the source line page.
- Evidence:
  - `eval/line_color_dataset_pair_20260610/comparison_sheet.jpg`
  - `eval/line_color_dataset_pair_20260610_lowdenoise/comparison_sheet.jpg`

Adding `AnimaEasyControlPatch` improved spatial preservation:

- Panel borders, speech bubbles, and character placement were much more stable.
- Color was applied, but the result was still closer to rough flat/gradient
  colorization than polished final dataset-quality coloring.
- Evidence:
  - `eval/line_color_dataset_pair_easycontrol_ip_20260610/comparison_sheet.jpg`
  - `eval/line_color_dataset_pair_easycontrol_ip_20260610/report.md`

## Root Cause

PE IP-Adapter is image-reference guidance through cross-attention. It can bias
style, palette, and visual identity, but it is not a spatial line-control
mechanism. Line-art colorization requires preserving local structure from the
input image, so a spatial conditioning path is required.

## Next Stage

The next reference-control track is:

1. Build or obtain paired metadata for the Wenaka/Anima IP-Adapter dataset.
2. Train the native SigLIP2 adapter stack:
   - `SigLIP2 image features`
   - `CrossLayerEncoder`
   - `TimeResampler`
   - `IPCrossAttn`
3. Load the trained checkpoint through the native ComfyUI SigLIP nodes.
4. Evaluate it against reference-control tasks separately from colorization.

For line-art colorization, use a combined stack:

- spatial control: `AnimaEasyControlPatch` or a dedicated line/colorize control
  checkpoint
- reference control: PE or SigLIP IP-Adapter

