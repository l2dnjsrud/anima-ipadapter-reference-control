# SigLIP Auto-Caption Runtime Evaluation: siglip_runtime_quality_20260612_c035_suite_v1

- Contact sheet: `eval/siglip_runtime_quality_20260612_c035_suite_v1/contact_sheet.jpg`
- Columns: reference / no_ip / siglip_kv_init_w14 / siglip_ref_retrieval_w14

Decision: `not_ready`

## Metric And Visual Summary

| variant | cases | mean cosine | no-IP baseline | mean uplift | improved rate | metric gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `siglip_kv_init_w14` | 32 | 0.7728 | 0.7436 | +0.0292 | 0.6563 | fail |
| `siglip_ref_retrieval_w14` | 32 | 0.8013 | 0.7436 | +0.0577 | 0.6563 | fail |

Visual audit:

- Best current SigLIP variant: `siglip_ref_retrieval_w14`
- Blank generated images: `0`
- Palette/costume/expression/framing acceptable: `31 / 32`
- Identity/distinctive trait acceptable: `16 / 32`
- Non-human/special trait acceptable: `0 / 1`
- Decision reason: c035 produces attractive manhwa images and often improves broad style, but it fails the metric gate and the stricter identity/distinctive-trait gate. The common failure is template collapse toward black long-haired wuxia characters, purple/night palace lighting, red-eye villain traits, or generic official/elder templates.

See also:

- `eval/siglip_runtime_quality_20260612_c035_suite_v1/pe_similarity_metrics.json`
- `eval/siglip_runtime_quality_20260612_c035_suite_v1/visual_audit.json`
- `eval/siglip_runtime_quality_20260612_c035_suite_v1/visual_audit.md`

| sample | selected attributes |
| --- | --- |
| auto00 | long black-haired wuxia swordsman, long flowing black hair, androgynous elegant warrior, stern serious expression, side profile portrait, black martial robe with red trim, night palace rooftop, dark night palace background, young clean-shaven warrior, pale purple-skinned villain, flowing robe in motion, folding fan in hand |
| auto01 | angry martial artist close-up, angry tense expression, young clean-shaven warrior, side profile portrait, androgynous elegant warrior, messy black hair, sharp fangs visible, drawn sword action pose, metal shoulder armor, tan traditional robe, outdoor martial arts courtyard, purple villain aura lighting |
| auto02 | old bearded martial arts master, bald old monk with prayer beads, elder male master, stern serious expression, red glowing demonic eye, side profile portrait, tan traditional robe, raised hand martial arts gesture, long flowing black hair, metal shoulder armor, outdoor martial arts courtyard, purple villain aura lighting |
| auto03 | red-haired noble woman in ornate dress, ornate red and gold palace dress, female noble court character, red gold indoor palace colors, calm seated expression, seated full-body indoor panel, flowing robe in motion, red hair and pale makeup, indoor throne hall background, red glowing demonic eye, folding fan in hand, old bearded martial arts master |
| auto04 | angry martial artist close-up, angry tense expression, side profile portrait, androgynous elegant warrior, young clean-shaven warrior, long flowing black hair, black martial robe with red trim, red glowing demonic eye, raised hand martial arts gesture, outdoor martial arts courtyard, sword hilt visible, bright daylight courtyard |
| auto05 | angry martial artist close-up, angry tense expression, old bearded martial arts master, male wuxia swordsman, upper body close-up portrait, black official hat, sharp fangs visible, messy black hair, black official hat and formal robe, raised hand martial arts gesture, purple villain aura lighting, outdoor martial arts courtyard |
| auto06 | middle-aged court official with black hat, stern serious expression, blue gray scholar robe and official hat, black official hat, black mustache and official face, upper body close-up portrait, androgynous elegant warrior, hair tied in topknot, pale purple-skinned villain, raised hand martial arts gesture, purple villain aura lighting, night palace rooftop |
| auto07 | green monster face with red glowing eye, angry tense expression, angry martial artist close-up, side profile portrait, green eerie cave lighting, androgynous elegant warrior, metal shoulder armor, old bearded martial arts master, dark armored robe, clenched fist foreground, messy black hair, night palace rooftop |
| auto08 | angry martial artist close-up, angry tense expression, androgynous elegant warrior, arm thrust forward action pose, raised hand martial arts gesture, long flowing black hair, young clean-shaven warrior, dark armored robe, folding fan in hand, sharp fangs visible, outdoor martial arts courtyard, purple villain aura lighting |
| auto09 | long black-haired wuxia swordsman, angry tense expression, androgynous elegant warrior, long flowing black hair, side profile portrait, young clean-shaven warrior, drawn sword action pose, black martial robe with red trim, sharp fangs visible, metal shoulder armor, night palace rooftop, cool blue gray palace lighting |
| auto10 | old bearded martial arts master, middle-aged court official with black hat, stern serious expression, elder male master, blue gray scholar robe and official hat, round scholar glasses, hair tied in topknot, upper body close-up portrait, pale purple-skinned villain, raised hand martial arts gesture, purple villain aura lighting, outdoor martial arts courtyard |
| auto11 | old bearded martial arts master, angry martial artist close-up, stern serious expression, elder male master, side profile portrait, sharp fangs visible, tan traditional robe, long flowing black hair, metal shoulder armor, raised hand martial arts gesture, outdoor martial arts courtyard, purple villain aura lighting |
| auto12 | androgynous elegant warrior, angry martial artist close-up, long flowing black hair, sharp eyebrows and stern eyes, side profile portrait, young clean-shaven warrior, black martial robe with red trim, sharp fangs visible, drawn sword action pose, metal shoulder armor, purple villain aura lighting, outdoor martial arts courtyard |
| auto13 | angry tense expression, angry martial artist close-up, androgynous elegant warrior, long flowing black hair, upper body close-up portrait, young clean-shaven warrior, black martial robe with red trim, drawn sword action pose, folding fan in hand, red glowing demonic eye, outdoor martial arts courtyard, dark night palace background |
| auto14 | side profile portrait, angry martial artist close-up, angry tense expression, androgynous elegant warrior, long flowing black hair, young clean-shaven warrior, black martial robe with red trim, drawn sword action pose, red glowing demonic eye, metal shoulder armor, outdoor martial arts courtyard, purple villain aura lighting |
| auto15 | angry martial artist close-up, angry tense expression, old bearded martial arts master, androgynous elegant warrior, long flowing black hair, side profile portrait, black martial robe with red trim, raised hand martial arts gesture, outdoor martial arts courtyard, sharp fangs visible, dark night palace background, sword hilt visible |
| auto16 | angry martial artist close-up, androgynous elegant warrior, angry tense expression, upper body close-up portrait, young clean-shaven warrior, long flowing black hair, drawn sword action pose, black martial robe with red trim, sharp fangs visible, metal shoulder armor, outdoor martial arts courtyard, purple villain aura lighting |
| auto17 | angry tense expression, side profile portrait, angry martial artist close-up, long flowing black hair, female noble court character, red glowing demonic eye, black mustache and official face, drawn sword action pose, folding fan in hand, night palace rooftop, black martial robe with red trim, misty blue moonlight |
| auto18 | calm seated expression, androgynous elegant warrior, angry martial artist close-up, seated full-body indoor panel, young clean-shaven warrior, long flowing black hair, drawn sword action pose, black martial robe with red trim, outdoor martial arts courtyard, folding fan in hand, pale purple-skinned villain, purple villain aura lighting |
| auto19 | angry martial artist close-up, angry tense expression, bald old monk with long white gray beard, upper body close-up portrait, human martial arts character, sharp fangs visible, hair tied in topknot, metal shoulder armor, raised hand martial arts gesture, purple villain aura lighting, purple villain robe, night palace rooftop |
| auto20 | angry martial artist close-up, red glowing demonic eye, angry tense expression, old bearded martial arts master, red prayer beads necklace, androgynous elegant warrior, long flowing black hair, upper body close-up portrait, tan traditional robe, purple villain aura lighting, raised hand martial arts gesture, outdoor martial arts courtyard |
| auto21 | stern serious expression, angry martial artist close-up, androgynous elegant warrior, side profile portrait, long flowing black hair, young clean-shaven warrior, black martial robe with red trim, sharp fangs visible, drawn sword action pose, sword hilt visible, outdoor martial arts courtyard, cool blue gray palace lighting |
| auto22 | old bearded martial arts master, stern serious expression, tan traditional robe, angry martial artist close-up, long flowing black hair, elder male master, upper body close-up portrait, flowing robe in motion, fur collar cloak, pale purple-skinned villain, wooden temple interior, bright daylight courtyard |
| auto23 | bald old monk with prayer beads, bald old monk with long white gray beard, stern serious expression, elder male master, upper body close-up portrait, green scholar robe, red glowing demonic eye, red prayer beads necklace, raised hand martial arts gesture, hair tied in topknot, outdoor martial arts courtyard, purple villain aura lighting |
| auto24 | angry martial artist close-up, angry tense expression, androgynous elegant warrior, hair tied in topknot, young clean-shaven warrior, upper body close-up portrait, black martial robe with red trim, raised hand martial arts gesture, sharp fangs visible, sword hilt visible, purple villain aura lighting, outdoor martial arts courtyard |
| auto25 | middle-aged court official with black hat, old bearded martial arts master, angry tense expression, androgynous elegant warrior, blue gray scholar robe and official hat, raised hand martial arts gesture, arm thrust forward action pose, outdoor martial arts courtyard, black official hat, hair tied in topknot, pale purple-skinned villain, purple villain aura lighting |
| auto26 | upper body close-up portrait, female noble court character, angry tense expression, red glowing demonic eye, angry martial artist close-up, night palace rooftop, metal shoulder armor, red hair and pale makeup, purple villain aura lighting, raised hand martial arts gesture, dark armored robe, young clean-shaven warrior |
| auto27 | stern serious expression, middle-aged court official with black hat, blue gray scholar robe and official hat, androgynous elegant warrior, red glowing demonic eye, side profile portrait, old bearded martial arts master, round scholar glasses, dark hair with ornate hairpin, raised hand martial arts gesture, cool blue gray palace lighting, night palace rooftop |
| auto28 | gold embroidered court robe, middle-aged court official with black hat, indoor throne hall background, old bearded martial arts master, seated full-body indoor panel, androgynous elegant warrior, angry tense expression, red gold indoor palace colors, long flowing black hair, flowing robe in motion, pale purple-skinned villain, black official hat |
| auto29 | androgynous elegant warrior, side profile portrait, stern serious expression, long black-haired wuxia swordsman, long flowing black hair, young clean-shaven warrior, sharp fangs visible, raised hand martial arts gesture, white flowing martial robe, metal shoulder armor, cool blue gray palace lighting, outdoor martial arts courtyard |
| auto30 | old bearded martial arts master, bald old monk with prayer beads, elder male master, stern serious expression, tan traditional robe, red prayer beads necklace, side profile portrait, pale purple-skinned villain, raised hand martial arts gesture, outdoor martial arts courtyard, long loose white hair, purple villain aura lighting |
| auto31 | middle-aged court official with black hat, blue gray scholar robe and official hat, black official hat, stern serious expression, long flowing black hair, human martial arts character, black mustache and official face, side profile portrait, raised hand martial arts gesture, red glowing demonic eye, outdoor martial arts courtyard, purple villain aura lighting |
