from __future__ import annotations

import importlib
import sys
from pathlib import Path

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _merge(module_name: str) -> None:
    try:
        if __package__ and __package__ in sys.modules:
            module = importlib.import_module(f".{module_name}", __package__)
        else:
            raise ModuleNotFoundError(module_name)
    except ModuleNotFoundError as error:
        if error.name != "folder_paths":
            module_dir = str(Path(__file__).resolve().parent)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as retry_error:
                if retry_error.name != "folder_paths":
                    raise
                return
        else:
            return
    NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(
        getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
    )


_merge("nodes")
_merge("native_pe")
_merge("native_siglip")
_merge("native_qwenvl")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
