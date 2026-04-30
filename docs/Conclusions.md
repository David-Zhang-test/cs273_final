# Conclusions from SAE Interpretation and Steering

## Main Conclusion

The SAE results do not support a single abstention core that appears early and then simply grows into a later category chooser. Instead, the representation reorganizes across layers:

1. Layers 7 and 11 are still mixed and relatively generic.
2. Layer 15 is a transition layer with strong geometric separation, but it is still dominated by mixed response-style features.
3. Layer 19 continues the transition and begins to sharpen refusal-related structure.
4. Layer 23 is the main turning point where category-specific refusal signatures become much clearer.
5. Layer 27 is the most specialized layer and the best steering target for subtype-level behavior shifts.

This means abstention is not a single stable binary gate. It is better described as a progressively reorganized feature space where category-specific signatures become sharper in later layers.

## What Counts as Core vs Category-Specific

For this document, the **core nodes** are the features that recur across many categories and behave like shared latent directions. The **category-specific nodes** are the features that most strongly distinguish one category from the others.

The layer-27 results show a shared backbone, but that backbone is not perfectly uniform across all labels. The strongest shared nodes are:

- `44833` - very strong across almost every category
- `80948` - strong across most categories, especially refusal-like categories
- `106628` - strong shared feature, but with especially high activation for Epistemic Refusal and Both-Sides Neutrality

Additional semi-shared nodes that recur across several categories:

- `24431`
- `6326`
- `45181`
- `40311`

## Per-Category Node Map

### 1. Full Compliance

- Core nodes: `44833`, `80948`, `106628`
- Category-specific nodes: `129304`, `32672`, `5506`
- Interpretation: compliance is not represented by a completely separate space; it shares the common backbone but is distinguished by compliance-weighted nodes such as `129304` and `32672`.

### 2. Caveat / Disclaimer Compliance

- Core nodes: `44833`, `80948`, `106628`
- Category-specific nodes: `9897`, `76568`, `24431`
- Interpretation: this category still uses the shared backbone, but `9897` and `76568` help mark the disclaimer-heavy style.

### 3. Preach / Lecturing Compliance

- Core nodes: `44833`, `80948`, `106628`
- Category-specific nodes: `9897`, `40311`, `76568`
- Interpretation: lecturing compliance is close to caveated compliance, but `9897` is especially prominent here.

### 4. Both-Sides Neutrality

- Core nodes: `44833`, `80948`, `106628`
- Category-specific nodes: `45181`, `83243`, `106628`
- Interpretation: this is one of the clearest mixed categories. It keeps the shared backbone but adds neutrality-specific weighting, especially through `45181` and the high activation of `106628`.

### 5. The Pivot / Deflection

- Core nodes: `44833`, `80948`, `106628`
- Category-specific nodes: `9897`, `24431`, `40311`
- Interpretation: pivoting reuses much of the shared space but shifts toward adjacent-answer style features rather than direct compliance or direct refusal.

### 6. Theory-Only (Partial Refusal)

- Core nodes: `44833`, `80948`, `106628`
- Category-specific nodes: `44833`, `76568`, `24431`
- Interpretation: theory-only behavior is strongly marked by `44833` and `76568`, which makes it look like a refined refusal mode rather than a separate mechanism.

### 7. Polite Full Refusal

- Core nodes: `44833`, `80948`, `106628`
- Category-specific nodes: `45181`, `78807`, `40311`
- Interpretation: polite refusal is one of the clearest subtype signatures. `45181` and `78807` are especially useful for steering toward or away from this category.

### 8. Epistemic Refusal

- Core nodes: `44833`, `80948`, `106628`
- Category-specific nodes: `40311`, `42366`, `82391`
- Interpretation: epistemic refusal is the strongest subtype-specific refusal signature in layer 27. `106628` and `80948` are shared, but `40311`, `42366`, and `82391` make the epistemic subtype stand out.

## Practical Steering Implication

The steering experiment suggests that layer 27 is better at changing refusal style than cleanly flipping refusal into compliance. In other words, it is a strong candidate for subtype steering, but not yet proof of a clean abstain-vs-comply gate.

That leads to the following operational conclusion:

1. Use the shared-core nodes (`44833`, `80948`, `106628`) to test whether the overall refusal tendency weakens.
2. Use the category-specific nodes to test whether the output shifts among refusal styles.
3. Treat layer 23 as a likely transition layer and layer 27 as the strongest subtype-crystallization layer.

## Short Final Summary

The model does not appear to have one single abstention core plus a later category chooser. Instead, it has a shared latent backbone and then progressively more category-specific nodes in the later layers. The most specialized structure appears at layer 27, and the clearest category-specific refusal nodes are:

- Polite Full Refusal: `45181`, `78807`, `40311`
- Both-Sides Neutrality: `45181`, `83243`, `106628`
- Theory-Only (Partial Refusal): `44833`, `76568`, `24431`
- Caveat / Disclaimer Compliance: `9897`, `76568`, `24431`
- Preach / Lecturing Compliance: `9897`, `40311`, `76568`
- Full Compliance: `129304`, `32672`, `5506`
- Epistemic Refusal: `40311`, `42366`, `82391`
- The Pivot / Deflection: `24431`, `40311`, `9897`
