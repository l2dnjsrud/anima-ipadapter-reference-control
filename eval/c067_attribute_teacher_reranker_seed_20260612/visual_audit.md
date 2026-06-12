# c067 Visual Audit

- Artifact: `attribute_review_sheet.jpg`
- Candidate rows: `72`
- Queries: `6`
- Scorer: `Qwen/Qwen3-VL-Embedding-2B`

## Observations

`direct_green_non_human_face` is not clean. Its top candidates include old human faces,
headwear/shadow-heavy close-ups, a red-eyed monk-like character, ordinary human faces,
tea cups, and green-tinted panels. The query is therefore still entangled with old-face,
red-eye, and green-object/background cues.

`red_glowing_eye`, `side_profile_silhouette`, and `beard_headwear_crop` are more useful
as review queues. Their top rows show plausible red-eye faces, profile faces, and older
bearded/headwear crops.

`background_object_green` behaves as a useful false-positive guard. It catches cups,
leaves, building decor, and green background panels that should not become direct-green
character positives.

## Decision

Do not train direct-green/non-human positives from c067 automatically. Use c067 as a
review queue and build explicit labels or a stronger captioning teacher before the next
encoder-side objective.
