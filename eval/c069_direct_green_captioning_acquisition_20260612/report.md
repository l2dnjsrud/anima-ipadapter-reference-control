# c069 Direct Green Captioning Acquisition

- Scanned images: `1563`
- Heldout rows used: `0`
- Candidate rows: `48`
- Direct-green target positives: `0`
- Useful non-human proxies: `2`
- Decision: `new_dataset_captioning_required`

The full local color pool was scanned beyond c067 top-k, but the reviewed top green/red queues are still dominated by background, objects, lighting, cups, leaves, and non-target color. Do not train encoder-side direct-green positives from this seed unless at least 4 confirmed target positives are collected.
