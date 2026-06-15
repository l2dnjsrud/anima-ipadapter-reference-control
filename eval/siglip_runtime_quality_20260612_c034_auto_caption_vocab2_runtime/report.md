# SigLIP Auto-Caption Runtime Evaluation: siglip_runtime_quality_20260612_c034_auto_caption_vocab2_runtime

- Contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/siglip_runtime_quality_20260612_c034_auto_caption_vocab2_runtime/contact_sheet.jpg`
- Columns: reference / no_ip / siglip_pe_space_w14 / siglip_pe_retrieval_w14
- Decision: `siglip_auto_caption_single_character_visual_pass_pe_metric_caveat`
- Note: Expanded attribute vocabulary run. The native SigLIP checkpoints preserve most single-character identity, palette, expression, and costume cues better than no-IP. The monster sample is visually closer even though PE pooled-cosine drops, so PE metrics are treated as auxiliary rather than the sole pass/fail signal.

## Variant Metrics

| variant | cases | mean cosine | mean no-IP cosine | mean uplift | improved rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| siglip_pe_retrieval_w14 | 8 | 0.8227 | 0.6775 | +0.1452 | 87.5% |
| siglip_pe_space_w14 | 8 | 0.7878 | 0.6775 | +0.1103 | 87.5% |

## Selected Attributes

| sample | selected attributes |
| --- | --- |
| auto00 | long black-haired wuxia swordsman, long flowing black hair, stern serious expression, side profile portrait, black martial robe with red trim, human martial arts character, dark night palace background |
| auto01 | angry martial artist close-up, angry tense expression, sharp eyebrows and stern eyes, upper body close-up portrait, human martial arts character, tan traditional robe, red gold indoor palace colors |
| auto02 | old bearded martial arts master, white gray beard and thick eyebrows, stern serious expression, human martial arts character, upper body close-up portrait, tan traditional robe, cool blue gray palace lighting |
| auto03 | red-haired noble woman in ornate dress, ornate red and gold palace dress, red gold indoor palace colors, calm seated expression, seated full-body indoor panel, red hair and pale makeup, human martial arts character |
| auto04 | angry martial artist close-up, angry tense expression, sharp eyebrows and stern eyes, side profile portrait, human martial arts character, black martial robe with red trim, dark night palace background |
| auto05 | angry martial artist close-up, angry tense expression, open screaming mouth, human martial arts character, upper body close-up portrait, black official hat and formal robe, dark night palace background |
| auto06 | middle-aged court official with black hat, stern serious expression, blue gray scholar robe and official hat, sharp eyebrows and stern eyes, upper body close-up portrait, human martial arts character, cool blue gray palace lighting |
| auto07 | green monster face with red glowing eye, angry tense expression, angry martial artist close-up, open screaming mouth, side profile portrait, dark armored robe, cool blue gray palace lighting |
