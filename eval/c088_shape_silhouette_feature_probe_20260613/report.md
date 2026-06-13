# c088 Shape/Silhouette Feature Probe

- Decision: `shape_signal_present_encoder_embedding_not_enough`
- Next direction: `distill_edge_silhouette_or_train_encoder_side_shape_checkpoint`

## Feature Summaries

| feature | cases | supported | support rate | source decision |
| --- | ---: | ---: | ---: | --- |
| `edge_projection_silhouette` | `11` | `6` | `0.5454545454545454` | `shape_silhouette_signal_viable` |
| `qwenvl` | `11` | `4` | `0.36363636363636365` | `embedding_signal_not_viable` |
| `siglip2` | `11` | `4` | `0.36363636363636365` | `embedding_signal_not_viable` |
| `pe` | `11` | `7` | `0.6363636363636364` | `embedding_signal_viable` |

## heldout07 Hard Case

| feature | best variant | uplift | margin | decision |
| --- | --- | ---: | ---: | --- |
| `edge_projection_silhouette` | `c087_expanded_crop_positive_w14` | `0.024618472899662236` | `0.014583258045160363` | `shape_signal_not_enough` |
| `qwenvl` | `no_ip` | `0.0` | `0.00960475206375122` | `embedding_signal_not_enough` |
| `siglip2` | `c086_hard_negative_w14` | `0.024952709674835205` | `0.024952709674835205` | `embedding_signal_not_enough` |
| `pe` | `c086_hard_negative_w14` | `0.07166695594787598` | `0.06360191106796265` | `embedding_signal_supports_supervised_objective` |

## Interpretation

c088 checks whether the reference-control failure is merely an adapter head issue or whether the image encoder feature space is missing shape/silhouette signal.
The edge/projection/silhouette metric and PE contain partial shape signal, but QwenVL and SigLIP2 do not reach the support threshold on this c087 hard-shape set.
Because the active native adapter path depends on QwenVL/SigLIP-like image embeddings, the next step should not repeat broad adapter continuation; it should either distill explicit shape features or train an encoder-side shape checkpoint.
