# SigLIP Workflow Validation 2026-06-12

## Scope

This validation checks the packaged SigLIP attribute-reference workflow path for ComfyUI. It uses the repo-local custom node loaded by the temporary ComfyUI server at `http://127.0.0.1:8116`.

## Static Tests

Command:

```bash
PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python -m pytest tests/test_comfyui_workflows.py tests/test_native_siglip.py -q
```

Result: pass, `20 passed`.

Evidence: `.omo/evidence/task-6-pytest.txt`

## Node Registration

ComfyUI `object_info` contains:

- `AnimaSigLIPIPAdapterLoader`
- `AnimaSigLIPEncodeImage`
- `AnimaSigLIPIPAdapterApply`

The `AnimaSigLIPIPAdapterLoader` model selector contains:

- `anima_siglip_ip_adapter_single_character_clean32_pe_retrieval_0128_20260611.safetensors`

Evidence: `.omo/evidence/task-6-object-info-check.txt`

## API Smoke

The c035 runtime evaluation executed the same native SigLIP API graph and produced nonblank images. The smoke sample:

- image: `eval/siglip_runtime_quality_20260612_c035_suite_v1/auto00_siglip_ref_retrieval_w14.png`
- pixel std: `63.273170471191406`
- nonblank: `true`

Evidence: `.omo/evidence/task-6-api-smoke-nonblank.txt`

## Failure Matrix

| Case | Expected | Observed | Evidence |
| --- | --- | --- | --- |
| PE-Core checkpoint selected in SigLIP loader | explicit family mismatch error | `AnimaSigLIPIPAdapterLoader` rejects it and says to use the PE loader | `.omo/evidence/task-6-wrong-family.txt` |
| missing reference image in eval manifest | nonzero failure before false success | `FileNotFoundError` for missing `ref_id` image | `.omo/evidence/task-5-c035-failure-mode.txt` |
| stale installed ComfyUI02 node copy | shape mismatch and failed history | old node rejected the newer SigLIP checkpoint; local repo symlink fixed it | `.omo/evidence/task-5-failed-history-probe.txt` |

## Result

ComfyUI integration is usable through the native `AnimaSigLIP*` node family when the server loads this repo's current custom node code. The existing generated quality is still governed by the c035 decision: API/UI execution works, but the model is not yet a high-quality trustworthy reference-control checkpoint.
