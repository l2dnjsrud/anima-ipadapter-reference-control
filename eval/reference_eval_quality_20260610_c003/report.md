# IP-Adapter Reference-Control Evaluation

**Status:** PASS
**Checkpoint:** `output/ckpt/anima_ip_adapter_quality_20260610.safetensors`
**Prompt:** masterpiece, best quality, score_7, safe. 1girl, solo, standing in a cafe, holding a coffee cup, looking at viewer, smile, soft lighting.
**Seeds:** [20260610, 20260611]
**Scales:** [0.5, 1.0, 1.5, 2.0]

## Thresholds

- Mean uplift: `0.03`
- Improved rate: `0.75`
- Min pixel std: `5.0`

## Summary

- Best scale: `1.0`
- Generated images: `40`
- Nonblank: `True`

## Commands

- Generation manifest: `output/bench/ip_adapter/reference_eval_quality_20260610_c003/manifest.json`
- Generation script: `output/bench/ip_adapter/reference_eval_quality_20260610_c003/run_eval.sh`
- Generation command coverage: `run_eval.sh` contains the exact no-IP and IP-scale command for every manifest job.
- Score command:

```bash
/home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python bench/ip_adapter/reference_eval.py score --manifest output/bench/ip_adapter/reference_eval_quality_20260610_c003/manifest.json --device cuda:0 --min-std 5 --mean-uplift-threshold 0.03 --improved-rate-threshold 0.75
```

| Scale | Cases | Mean IP Cos | Mean No-IP Cos | Mean Uplift | Improved Rate |
|---|---:|---:|---:|---:|---:|
| 0.5 | 8 | 0.7273 | 0.7014 | 0.0259 | 75.00% |
| 1 | 8 | 0.7988 | 0.7014 | 0.0974 | 87.50% |
| 1.5 | 8 | 0.7517 | 0.7014 | 0.0503 | 75.00% |
| 2 | 8 | 0.7645 | 0.7014 | 0.0632 | 62.50% |

## References

- `ref00`: `post_image_dataset/resized/001-100/SG-083/mrcolor_panel_style_v4_split_edge_strict_probe_00494_mrcolor_panel_style_v4_split_candidate_mrcolor_panel_style_v4_candidate_02718_SG-083-10_page_719x957_s01.png`
- `ref01`: `post_image_dataset/resized/001-100/SG-034/mrcolor_panel_style_v4_split_edge_strict_probe_00219_mrcolor_panel_style_v4_split_candidate_mrcolor_panel_style_v4_candidate_01056_SG-034-10_page_2046x2874_s01.png`
- `ref02`: `post_image_dataset/resized/101-200/SG-169/mrcolor_panel_style_v4_split_edge_strict_probe_00994_mrcolor_panel_style_v4_split_candidate_mrcolor_panel_style_v4_candidate_05709_SG-169-09_page_967x818_s01.png`
- `ref03`: `post_image_dataset/resized/101-200/SG-126/mrcolor_panel_style_v4_split_edge_strict_probe_00761_mrcolor_panel_style_v4_split_candidate_mrcolor_panel_style_v4_candidate_04287_SG-126-14_page_2072x2889_s02.png`

## Limitations

- PE pooled-cosine measures visual/reference proximity in the same PE-Core family used by this local IP-Adapter path.
- This dataset currently uses path-derived groups for generic layout/style captions, so the result is layout/style reference-control rather than verified character identity recovery.
