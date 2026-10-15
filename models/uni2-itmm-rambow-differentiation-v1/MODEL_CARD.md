# Model card: uni2-itmm-rambow-differentiation-v1

## Intended use

Clustering of tumor tiles from H&E whole-slide images of in-transit melanoma metastases into clusters associated
with high or low Rambow differentiation, and slide-level prediction of high Rambow differentiation.
For non-commercial academic research only.

## Inputs

- Tiles of 0.125 mm, resized to 224 px.
- Features: MahmoodLab/UNI2-h (revision `d517a8dd47902dd7c308b3c36f63bce47e7b9a43`).
- Tumor filter: MahmoodLab/CONCH (revision `f9ca9f877171a28ade80228fb195ac5d79003357`),
  prompts v1.

Results are only meaningful with these backbones, revisions and tile size. Staining and scanner differences can
shift cluster assignments.

## Training

- 33 slides (21 high, 12 low; high means
  Rambow differentiation >= 0.5), up to 1000 tiles per slide, 14,764 tumor tiles.
- 4 PLS components, 8 k-means clusters.

## Cluster associations

Mann-Whitney U test of per-slide cluster presence between high and low training slides
(alpha = 0.05). The p-values describe the training cohort, not new data.

| Cluster | Association | p-value |
|---|---|---|
| 0 | low | 0.0185 |
| 1 | none | 0.303 |
| 2 | none | 0.598 |
| 3 | none | 0.779 |
| 4 | none | 0.329 |
| 5 | high | 0.0453 |
| 6 | none | 0.105 |
| 7 | none | 0.0644 |

## Slide model

Logistic regression (C = 1.0) on standardised degree-2 polynomial features of presence_high and presence_low. `rd_high = rd_high_proba >= 0.5`.

## Validation

```json
{
  "essen_itmm_in_sample_auc": 0.8353174603174603,
  "essen_itmm_loso_auc": 0.6865079365079365,
  "lsp_itmm_external_auc": 0.5380434782608696,
  "note": "Essen: 33 training slides (leave-one-slide-out refits the slide model only, clusters fixed). LSP: 39 external slides with binary labels."
}
```

## Licence

This model is derived from UNI-2 and CONCH outputs, whose terms (CC-BY-NC-ND 4.0) prohibit commercial use of
derivatives. It must not be used commercially.
