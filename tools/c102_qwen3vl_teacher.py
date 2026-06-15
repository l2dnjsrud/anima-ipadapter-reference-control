from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Qwen3VLTeacherConfig:
    model_path: Path = Path("/data/ai/models/LLM/Qwen-VL/Qwen3-VL-8B-Instruct")
    min_pixels: int = 224 * 224
    max_pixels: int = 512 * 512
    max_new_tokens: int = 64


class Qwen3VLTeacher:
    def __init__(self, config: Qwen3VLTeacherConfig = Qwen3VLTeacherConfig()) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            local_files_only=True,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto",
        )
        self._max_new_tokens = config.max_new_tokens

    def ask(self, image_path: str, prompt: str) -> str:
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda:0")
        with self._torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )
        trimmed = [output[len(input_ids) :] for input_ids, output in zip(inputs.input_ids, output_ids)]
        return str(
            self._processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        )
