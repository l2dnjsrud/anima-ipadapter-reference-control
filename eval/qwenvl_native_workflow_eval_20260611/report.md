# QwenVL Native ComfyUI Node Registration

- Server: isolated ComfyUI on `127.0.0.1:8117`
- Custom node: `/home/wktwin/anima-ipadapter-reference-control`
- Result: QwenVL nodes import and appear in ComfyUI `object_info`.

Registered nodes:

- `AnimaQwenVLIPAdapterLoader` -> `ANIMA_QWENVL_IPADAPTER`
- `AnimaQwenVLEncodeImage` -> `QWENVL_EMBEDDING`
- `AnimaQwenVLIPAdapterApply` -> `MODEL`

Evidence:

- `object_info_AnimaQwenVLIPAdapterLoader.json`
- `object_info_AnimaQwenVLEncodeImage.json`
- `object_info_AnimaQwenVLIPAdapterApply.json`

Status: this proves the native ComfyUI QwenVL node surface is visible. It is not
yet a quality or generation pass; that requires a trained QwenVL checkpoint and
the same no-IP/adapter contact-sheet evaluation used for the SigLIP branch.
