from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SigLIPCheckpointError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason
