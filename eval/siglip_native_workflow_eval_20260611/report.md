# SigLIP Native Workflow Eval 20260611

## Verdict

Blocked before SigLIP image generation.

The native SigLIP UI workflow and API prompt shape are now in place, and the
live ComfyUI API at `http://127.0.0.1:8102` can run the same base Anima graph
without IP-Adapter. The SigLIP IP-Adapter branch is blocked because the trained
pilot checkpoint is not visible in `AnimaSigLIPIPAdapterLoader`'s
`ipadapter_name` selector.

This run does not prove visual reference-control quality.

## What Passed

- Added UI workflow: `workflows/anima_ipadapter_siglip_native_reference.json`.
- The workflow is a normal ComfyUI graph, not `AnimaIPAdapterGenerate`.
- The loader uses selector name
  `anima_siglip_ip_adapter_pilot_20260610.safetensors`, not a raw path.
- The apply node feeds the patched `MODEL` to both `CFGGuider` and
  `BasicScheduler`.
- Focused workflow tests passed:
  `PYTHONPATH=/tmp/anima-ipadapter-testdeps /data/ai/comfyui02/.venv/bin/python -m pytest tests/test_comfyui_workflows.py -q`
- Result: `8 passed in 1.17s`.

## API Result

SigLIP prompt:

```text
POST http://127.0.0.1:8102/prompt
prompt: eval/siglip_native_workflow_eval_20260611/api_prompt_siglip_smoke.json
HTTP: 400 Bad Request
```

ComfyUI returned:

```text
prompt_outputs_failed_validation
AnimaSigLIPIPAdapterLoader 2
value_not_in_list
ipadapter_name: 'anima_siglip_ip_adapter_pilot_20260610.safetensors'
```

The live selector only lists:

```text
anima_ip_adapter_quality_20260610.safetensors
ip-adapter-faceid-plusv2_sdxl.bin
ip-adapter-plus_sd15.safetensors
ip-adapter-plus_sdxl_vit-h.safetensors
```

Non-sudo copy attempt:

```text
install -m 0644 checkpoints/anima_siglip_ip_adapter_pilot_20260610.safetensors /data/ai/models/ipadapter/anima_siglip_ip_adapter_pilot_20260610.safetensors
install: cannot create regular file ... Permission denied
```

No-IP smoke:

```text
POST http://127.0.0.1:8102/prompt
prompt: eval/siglip_native_workflow_eval_20260611/api_prompt_no_ip_smoke.json
prompt_id: 59348a21-244a-4534-9799-5d1c5bd4cb8b
status: success
image: eval/siglip_native_workflow_eval_20260611/no_ip_smoke_view_api.png
pixel stddev: [65.1568, 67.5347, 67.1537]
```

So the base ComfyUI API path is working; the blocker is specifically SigLIP
checkpoint visibility.

## Next Required Action

Install the pilot checkpoint where live ComfyUI can see it:

```bash
sudo install -m 0644 \
  /home/wktwin/anima-ipadapter-reference-control/checkpoints/anima_siglip_ip_adapter_pilot_20260610.safetensors \
  /data/ai/models/ipadapter/anima_siglip_ip_adapter_pilot_20260610.safetensors
```

Then restart or refresh ComfyUI model paths, confirm the selector lists
`anima_siglip_ip_adapter_pilot_20260610.safetensors`, and rerun the SigLIP
prompt. Only after that should we generate the SigLIP/no-IP/PE contact sheet and
judge visual quality.
