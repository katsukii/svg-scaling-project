# U-Shape Hypothesis Analysis: Token Length vs. Completion Rate

**Date**: 2026-04-28
**Context**: Section 5.6 supporting material
**Data source**: `results/samples_size_comparison/` (125 samples: 5 sizes x 25 samples each)
**Raw data**: `results/samples_size_comparison/token_length_analysis.json`

---

## 1. Background

Across five model sizes (Tiny to XL), completion rate and render rate exhibit a
non-monotonic (U-shaped) pattern, with Medium at the peak:

| Size   | Params | Val Loss | Completion | Render |
|--------|--------|----------|------------|--------|
| Tiny   | 1.8M   | 0.471    | 76%        | 48%    |
| Small  | 4.2M   | 0.417    | 68%        | 52%    |
| Medium | 13.8M  | 0.423    | **96%**    | **76%** |
| Large  | 35.7M  | 0.410    | 60%        | 48%    |
| XL     | 91.2M  | 0.381    | 72%        | 36%    |

### Hypothesis Under Test

> Larger models learn complex patterns from training data (long paths, detailed
> shapes). During generation, they attempt to reproduce these patterns, resulting
> in higher token counts. Consequently, they are more likely to exhaust
> `max_new_tokens` (4096) before emitting `</svg>`, lowering completion rate.

This predicts: (a) mean token length increases monotonically with model size,
and (b) incomplete samples from larger models are at the token limit.

---

## 2. Token Length Statistics

### 2.1 Complete Samples Only

| Size   | N  | Mean  | Median | P25  | P75   | P95   | Max  |
|--------|----|-------|--------|------|-------|-------|------|
| Tiny   | 19 | 1,640 | 1,265  | 763  | 2,522 | 3,388 | 3,400 |
| Small  | 17 | 1,941 | 2,069  | 1,102| 2,489 | 3,368 | 3,602 |
| Medium | 24 | 1,437 | 1,138  | 968  | 1,598 | 3,502 | 3,636 |
| Large  | 15 | 1,628 | 1,425  | 817  | 2,217 | 3,018 | 3,405 |
| XL     | 18 | **1,221** | **917** | 473 | 1,483 | 3,254 | 3,783 |

**Key finding**: XL produces the *shortest* complete outputs (mean 1,221), not
the longest. The ordering is Small > Tiny > Large > Medium > XL — no monotonic
relationship with model size.

### 2.2 Incomplete Samples Only

| Size   | N  | Mean  | Median | Min  | Max   |
|--------|----|-------|--------|------|-------|
| Tiny   | 6  | 3,444 | 4,098  | 86   | 4,153 |
| Small  | 8  | 4,124 | 4,115  | 4,098| 4,181 |
| Medium | 1  | 4,098 | 4,098  | 4,098| 4,098 |
| Large  | 10 | 4,132 | 4,132  | 4,098| 4,181 |
| XL     | 7  | 4,120 | 4,098  | 4,098| 4,181 |

### 2.3 All Samples

| Size   | N  | Mean  | Median |
|--------|----|-------|--------|
| Tiny   | 25 | 2,073 | 1,852  |
| Small  | 25 | 2,639 | 2,489  |
| Medium | 25 | 1,544 | 1,172  |
| Large  | 25 | 2,630 | 2,842  |
| XL     | 25 | 2,032 | 1,235  |

---

## 3. Incomplete Sample Analysis

### 3.1 Truncation at Token Limit

31 out of 32 incomplete samples (96.9%) have token lengths >= 4,000, confirming
that truncation at `max_new_tokens` is the dominant failure mode:

| Size   | Incomplete | >= 4,000 | Fraction |
|--------|-----------|----------|----------|
| Tiny   | 6         | 5        | 83%      |
| Small  | 8         | 8        | 100%     |
| Medium | 1         | 1        | 100%     |
| Large  | 10        | 10       | 100%     |
| XL     | 7         | 7        | 100%     |

The single outlier (Tiny, face_partial sample_4, 86 tokens) is a structural
collapse where the model broke mid-tag, a different failure mode from truncation.

### 3.2 Incompletion Rate Does Not Increase Monotonically

```
Medium (4%) < Tiny (24%) < XL (28%) < Small (32%) < Large (40%)
```

Large has the *highest* incompletion rate, not XL. The pattern is an inverted-U
(worst at Large), not a monotonic increase with model capacity.

---

## 4. Hypothesis Assessment

### Verdict: **(C) Contradicted**

The specific causal mechanism proposed — "larger models generate more tokens,
therefore hit the limit more often" — is **not supported** by the data.

| Hypothesis element                          | Evidence         | Status       |
|--------------------------------------------|------------------|--------------|
| XL learns more complex patterns            | Not directly measurable | Inconclusive |
| Larger models generate longer sequences    | XL complete mean = 1,221 (shortest) | **Contradicted** |
| Incomplete = hit `max_new_tokens` limit    | 31/32 at limit   | **Supported** |
| Incompletion rate increases with size      | Large > XL; non-monotonic | **Contradicted** |

### What the Data Actually Shows

The token length distribution is **bimodal** across all model sizes:

1. **Convergent mode**: The model produces a structurally valid SVG and
   terminates with `</svg>`. When this happens, larger models tend to be *more
   concise* (XL mean 1,221 vs Tiny mean 1,640).

2. **Degenerate mode**: The model fails to converge on a closing structure and
   continues generating until hitting the token limit (~4,098). This happens
   uniformly across all sizes — the token length at failure is always at the
   ceiling.

The critical factor is the **probability of entering the convergent mode**, not
the length of generation. Medium achieves 96% convergence; both smaller models
(insufficient capacity for reliable structure) and larger models (possible
underfitting relative to capacity) converge less reliably.

---

## 5. Revised Hypothesis for Section 5.6

The original hypothesis should be replaced with the following evidence-based
account:

> Generation exhibits a bimodal pattern: samples either converge to a complete
> SVG (often shorter for larger models) or degenerate until hitting the 4,096
> token limit, regardless of model size. The key differentiator is the
> probability of convergence, not the length of generated output.
>
> Medium (13.8M) achieves the highest convergence rate (96%), suggesting it
> occupies a sweet spot where model capacity is sufficient for reliable SVG
> structure but not so large as to be undertrained on our fixed data budget.
> Both smaller models (capacity-limited) and larger models (data-limited
> relative to their capacity) show lower convergence rates, producing the
> observed non-monotonic pattern.
>
> This finding has a practical implication: scaling validation loss alone does
> not predict downstream generation quality. While µP scaling law accurately
> predicts loss reduction (R² = 0.76), the generation success metrics follow a
> different dynamic governed by the capacity-data balance.

### Suggested phrasing for the report

> "We initially hypothesized that larger models generate longer sequences due to
> learning more complex SVG patterns, leading to truncation at the token limit.
> However, our token-level analysis reveals the opposite: XL produces the
> shortest complete outputs (mean 1,221 tokens vs. 1,640 for Tiny). Instead,
> incomplete samples uniformly hit the 4,096 token ceiling regardless of model
> size (31/32 cases). The non-monotonic completion rate is better explained by a
> bimodal generation dynamic — models either converge to a valid SVG or
> degenerate until truncation — with Medium (13.8M) achieving the highest
> convergence probability (96%)."
