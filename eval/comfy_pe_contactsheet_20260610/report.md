# IP-Adapter Reference-Control Evaluation

**Status:** PASS
**Checkpoint:** `/data/ai/models/ipadapter/anima_ip_adapter_quality_20260610.safetensors`
**Prompt:** masterpiece, best quality, score_7, safe. 1girl, solo, standing in a cafe, holding a coffee cup, looking at viewer, smile, soft lighting.
**Seeds:** [20260610]
**Scales:** [0.5, 1.0, 1.5, 2.0]

## Thresholds

- Mean uplift: `0.03`
- Improved rate: `0.75`
- Min pixel std: `5.0`

## Summary

- Best scale: `1.0`
- Generated images: `5`
- Nonblank: `True`

## Commands

- Generation manifest: `/home/wktwin/anima-ipadapter-reference-control/eval/comfy_pe_contactsheet_20260610/comfy_manifest.json`
- Generation channel: live ComfyUI HTTP API at `127.0.0.1:8102`.
- API run summary: `/home/wktwin/anima-ipadapter-reference-control/eval/comfy_pe_contactsheet_20260610/api_summary.json`
- Generation command coverage: `comfy_manifest.json` records each prompt id, output image, ref id, seed, scale, and image size.
- Score command:

```bash
/home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python bench/ip_adapter/reference_eval.py score --manifest /home/wktwin/anima-ipadapter-reference-control/eval/comfy_pe_contactsheet_20260610/comfy_manifest.json --device cuda:0 --min-std 5 --mean-uplift-threshold 0.03 --improved-rate-threshold 0.75
```

| Scale | Cases | Mean IP Cos | Mean No-IP Cos | Mean Uplift | Improved Rate |
|---|---:|---:|---:|---:|---:|
| 0.5 | 1 | 0.8017 | 0.7134 | 0.0882 | 100.00% |
| 1 | 1 | 0.8716 | 0.7134 | 0.1582 | 100.00% |
| 1.5 | 1 | 0.7680 | 0.7134 | 0.0546 | 100.00% |
| 2 | 1 | 0.8240 | 0.7134 | 0.1105 | 100.00% |

## References

- `ref03`: `post_image_dataset/resized/101-200/SG-126/mrcolor_panel_style_v4_split_edge_strict_probe_00761_mrcolor_panel_style_v4_split_candidate_mrcolor_panel_style_v4_candidate_04287_SG-126-14_page_2072x2889_s02.png`

## Limitations

- PE pooled-cosine measures visual/reference proximity in the same PE-Core family used by this local IP-Adapter path.
- This dataset currently uses path-derived groups for generic layout/style captions, so the result is layout/style reference-control rather than verified character identity recovery.
