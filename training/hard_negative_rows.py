from __future__ import annotations

from training.siglip_smoke_types import PairRow


def explicit_negative_or_fallback(row: PairRow, fallback: PairRow) -> PairRow:
    if row.neg_id is None:
        return fallback
    return PairRow(ref_id=row.neg_id, tgt_id=row.tgt_id, prompt=row.prompt)
