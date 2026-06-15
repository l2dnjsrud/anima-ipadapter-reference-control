from __future__ import annotations

import json
from pathlib import Path

import torch

from tools.score_auto_caption_qwenvl_metrics import score_auto_caption_summary


class FakeEmbedder:
    def __init__(self, embeddings: dict[str, torch.Tensor]) -> None:
        self._embeddings = embeddings

    def encode_image(self, image_path: Path) -> torch.Tensor:
        return self._embeddings[image_path.name]


def test_score_auto_caption_summary_compares_variants_to_no_ip(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    out_dir = tmp_path / "out"
    ref_dir = data_root / "refs"
    ref_dir.mkdir(parents=True)
    out_dir.mkdir()
    for path in (
        ref_dir / "a.jpg",
        out_dir / "a_no_ip.png",
        out_dir / "a_siglip_kv_init_w14.png",
        out_dir / "a_siglip_ref_retrieval_w14.png",
    ):
        path.write_bytes(b"placeholder")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "samples": [{"label": "auto00", "ref_id": "refs/a"}],
                "variants": [
                    {"label": "no_ip", "checkpoint": None, "weight": 0.0},
                    {"label": "siglip_kv_init_w14", "checkpoint": "kv", "weight": 1.4},
                    {"label": "siglip_ref_retrieval_w14", "checkpoint": "ref", "weight": 1.4},
                ],
                "results": {
                    "auto00_no_ip": {"image": str(out_dir / "a_no_ip.png")},
                    "auto00_siglip_kv_init_w14": {
                        "image": str(out_dir / "a_siglip_kv_init_w14.png")
                    },
                    "auto00_siglip_ref_retrieval_w14": {
                        "image": str(out_dir / "a_siglip_ref_retrieval_w14.png")
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    metrics = score_auto_caption_summary(
        summary_path,
        data_root=data_root,
        embedder=FakeEmbedder(
            {
                "a.jpg": torch.tensor([1.0, 0.0]),
                "a_no_ip.png": torch.tensor([0.0, 1.0]),
                "a_siglip_kv_init_w14.png": torch.tensor([1.0, 0.0]),
                "a_siglip_ref_retrieval_w14.png": torch.tensor([0.0, 1.0]),
            }
        ),
    )

    summaries = {
        str(item["variant"]): item
        for item in metrics["variant_summaries"]
        if isinstance(item, dict)
    }
    assert summaries["siglip_kv_init_w14"]["mean_uplift"] == 1.0
    assert summaries["siglip_kv_init_w14"]["improved_rate"] == 1.0
    assert summaries["siglip_ref_retrieval_w14"]["mean_uplift"] == 0.0
    assert summaries["siglip_ref_retrieval_w14"]["improved_rate"] == 0.0
