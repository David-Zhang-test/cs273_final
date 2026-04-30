# SAE Sweep, Interpretation, and Steering Notes

## 1) What we do with SAE in this project

We use SAEs in three stages:

1. Sweep across layers
- Goal: find where refusal/abstention-related behavior is most separable in SAE feature space.

2. Interpret top features
- Goal: understand what top SAE features represent by checking which judge categories activate them.
.json`, `category_feature_profiles.json`, `interpretation_summary.json`.

3. Steer behavior via SAE features
- Goal: test causal control by scaling/zeroing high-impact features.


## 2) How we do sweeping, and what the current data says

### Metrics used in the sweep

- cramers_v: effect-size style association score between category structure and feature variance.
- f_statistic: ANOVA-like statistic computed from between/within quantities.
- representational_distance: mean pairwise centroid distance between category means, normalized by feature spread.

### Current sweep results (from `saved_results/sae_sweep/layer_sweep_summary.csv`)

| Layer | Cramer's V | F-stat | Rep. Distance |
|---|---:|---:|---:|
| 3  | 0.1884 | 7.9487  | 88.4230  |
| 7  | 0.1627 | 50.7323 | 131.0258 |
| 11 | 0.1609 | 62.1221 | 137.6346 |
| 15 | 0.1643 | 92.1032 | 138.7732 |
| 19 | 0.1515 | 92.8195 | 121.0824 |
| 23 | 0.1399 | 90.7472 | 106.8589 |
| 27 | 0.1362 | 95.2772 | 93.7260  |

`best_layer` is 27 (based on max `f_statistic`).



### Where abstention behavior likely forms

From the metric trend, abstention/refusal appears to form in stages:

1. Early emergence (layer 3)
- Highest `cramers_v` suggests abstention-related category structure is already present early.

2. Mid-layer consolidation (layers 11-15)
- `representational_distance` peaks at layer 15, indicating category manifolds are farthest apart in this region.
- This is a strong candidate for where abstention modes become most organized in representation space.

3. Late policy/readout sharpening (layers 23-27)
- `f_statistic` increases to a maximum at layer 27, suggesting stronger between-vs-within category separation at readout.
- Layer 27 is selected by current F-stat criterion, consistent with a late decision/readout stage.

Practical conclusion:
- Abstention behavior likely starts forming early, becomes maximally structured around layer 15, and is strongly sharpened for final behavior selection by layer 27.



## 3) What we do to interpret and steer

### Interpretation workflow

Using `src/sae_analysis/feature_interpreter.py`:

1. Load top-K spread features from chosen layer sweep summary (default layer 15).
2. Parse judge categories from `judge_response`.
3. Encode activations into SAE feature space.
4. Compute per-feature, per-category statistics:
- mean activation
- standard deviation
- max activation
- activation rate (fraction > 0)
5. Build category-to-feature profiles and rank discriminative features.
6. Save structured artifacts for manual semantic labeling of features.

How to use this practically:
- Identify features with high activation in refusal categories and low activation in compliance categories.
- Manually inspect prompts that maximally activate those features to assign semantic interpretations (for example caution, policy-trigger, uncertainty framing).

### Steering workflow

Using `src/sae_analysis/steering_intervention.py` and `src/sae_analysis/steering.py`:

1. Choose a target layer and feature set (typically top refusal/abstention-related features from interpretation).
2. Apply interventions in feature space:
- zeroing (`scale = 0.0`) to suppress
- attenuation (`scale = 0.5`) to weaken
- amplification (`scale = 2.0`) to strengthen
3. Decode back to residual stream (or inject decoder direction via hook in live generation).
4. Compare steered vs original responses under the same prompts.
5. Re-judge outputs to measure category shift (for example compliance -> refusal or refusal -> compliance).

Recommended next steering experiment design:

- Compare three target layers directly: 3, 15, 27.
- Use same prompt subset and same judge pipeline.
- Report:
  - category transition matrix
  - refusal-rate delta
  - abstention subtype deltas (epistemic, polite refusal, neutrality, pivot)
- This will test whether layer 15 (representation peak) or layer 27 (readout peak) gives stronger causal control.

## Short answer summary

- SAE is used to sweep layers, interpret high-impact latent features, and steer behavior by intervening on those features.
- Current data suggests abstention signal emerges early (layer 3), is most geometrically organized around layer 15, and is sharpened for final decision/readout by layer 27.
- Interpretation maps features to categories; steering tests causality by scaling/zeroing those features and measuring judged output shifts.
