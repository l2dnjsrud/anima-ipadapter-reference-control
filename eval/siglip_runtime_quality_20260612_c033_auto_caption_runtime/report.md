# SigLIP Auto-Caption Runtime Evaluation: siglip_runtime_quality_20260612_c033_auto_caption_runtime

- Contact sheet: `/home/wktwin/anima-ipadapter-reference-control/eval/siglip_runtime_quality_20260612_c033_auto_caption_runtime/contact_sheet.jpg`
- Columns: reference / no_ip / siglip_pe_space_w14 / siglip_pe_retrieval_w14
- Decision: `siglip_auto_caption_vocab1_partial_visual_pass`
- Note: Initial auto-caption vocabulary run. Several references improve over no-IP, but the female and monster samples are visibly under-described, so this run is kept as a partial baseline.

## Variant Metrics

| variant | cases | mean cosine | mean no-IP cosine | mean uplift | improved rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| siglip_pe_retrieval_w14 | 8 | 0.7891 | 0.6682 | +0.1209 | 87.5% |
| siglip_pe_space_w14 | 8 | 0.7644 | 0.6682 | +0.0963 | 62.5% |

## Selected Attributes

| sample | selected attributes |
| --- | --- |
| auto00 | long black-haired wuxia swordsman, long flowing black hair, stern serious expression, side profile portrait, black martial robe with red trim, dark night palace background |
| auto01 | angry martial artist close-up, angry tense expression, sharp eyebrows and stern eyes, upper body close-up portrait, tan traditional robe, warm orange firelit background |
| auto02 | old bearded martial arts master, white gray beard and thick eyebrows, stern serious expression, upper body close-up portrait, tan traditional robe, cool blue gray palace lighting |
| auto03 | tan traditional robe, angry tense expression, upper body close-up portrait, angry martial artist close-up, open screaming mouth, dark night palace background |
| auto04 | angry martial artist close-up, angry tense expression, sharp eyebrows and stern eyes, side profile portrait, black martial robe with red trim, dark night palace background |
| auto05 | angry martial artist close-up, angry tense expression, open screaming mouth, upper body close-up portrait, blue gray scholar robe and official hat, dark night palace background |
| auto06 | angry martial artist close-up, stern serious expression, blue gray scholar robe and official hat, sharp eyebrows and stern eyes, upper body close-up portrait, cool blue gray palace lighting |
| auto07 | angry tense expression, angry martial artist close-up, open screaming mouth, side profile portrait, dark armored robe, cool blue gray palace lighting |
