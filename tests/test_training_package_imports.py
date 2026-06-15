from __future__ import annotations

import importlib.util
from pathlib import Path


def test_local_training_package_shadows_site_package() -> None:
    spec = importlib.util.find_spec("training.qwenvl_contrastive_smoke")

    assert spec is not None
    assert spec.origin is not None
    assert Path(spec.origin).parts[-2:] == ("training", "qwenvl_contrastive_smoke.py")
