# V13 RMPS: Reliability-Monotone Part Safety

## Decision

V12 is rejected as the module-one successor. V13 returns to V11's background-
suppressed part-aligned route and adds one conservative operation: low-
reliability parts attenuate only their own residual contribution.

V13 does not use distillation, a baseline teacher, module two, a candidate
branch, a new loss, or a new learned parameter. It is trained from the MAE
pretrain exactly like V8/V11.

## Why V12 failed

V12 multiplied the full route map of a part by a positive-evidence
preservation scale. The scale was computed from positive tokens but also
amplified that part's background tokens. In the targeted paired diagnostics,
the scale was larger on failed frames and had almost no relationship with the
marginal IoU benefit of VDRM. The learned LayerScale alpha then compensated
most of the global magnitude increase, so V12 changed optimization without
restoring the intended useful residual.

The formal V12 AUC was below V8 on all five evaluated datasets. The next
design must therefore be a clean V11-derived safety mechanism, not another
positive-mass compensation.

## V13 operation

Let `r_k` be the existing calibrated reliability of template part `k`. V11's
residual contribution for that part is

```text
w_v11(k, i) = route(k, i) * retention(k, i) * r_k
```

V13 introduces the detached safety factor

```text
s(r_k) = min(1, 2 * stop_gradient(r_k))
w_v13(k, i) = w_v11(k, i) * s(r_k)
```

The pivot `0.5` is the existing sigmoid classifier boundary, not a tuned
threshold. The factor has four useful invariants:

1. `r_k >= 0.5` gives `s(r_k) = 1`, so the part is exactly V11.
2. `r_k < 0.5` gives an effective reliability of `2 * r_k^2`.
3. `r * s(r)` is continuous and monotone non-decreasing.
4. `0 <= s(r) <= 1`, so V13 can never amplify V11's residual.

The factor is per part. One occluded or confused part cannot suppress another
visible, reliable part. Detaching the factor prevents tracking loss from
learning a shortcut through the safeguard; the existing reliability and
ranking losses remain responsible for calibration.

## Evidence that the safeguard activates selectively

The V12 targeted CSV files were replayed offline using the V13 formula. Across
the registered UAV123 and DTB70 sequences and both anchor modes (rows without
a valid GT box are excluded):

| VDRM status | Frames | Frames with any `r < 0.5` | Low-reliability part fraction | Mean safety factor |
| --- | ---: | ---: | ---: | ---: |
| correct | 9,162 | 12.55% | 6.84% | 0.9891 |
| failed | 1,386 | 82.90% | 68.25% | 0.8047 |
| ambiguous | 352 | 51.42% | 36.01% | 0.9158 |

This is the intended selectivity: correct frames remain close to V11, while
failed frames receive materially stronger attenuation. Reliability was not a
good predictor of marginal VDRM benefit, so V13 deliberately does not use it
as a general on/off gate for otherwise reliable parts.

## Controlled ablation

The explanatory chain is:

1. Baseline: no VDRM.
2. V8 PAR: part-aligned routing.
3. V11 BSPAR: V8 plus background route suppression.
4. V13 RMPS: V11 plus low-reliability per-part safety.

V12 remains a documented negative control for global positive-mass
compensation. No other training hyperparameter changes between V11 and V13.

## Diagnostics

Training logs and paired CSV/JSON outputs expose:

- `part_reliability_safety_factor_mean/min/max`
- `part_reliability_suppressed_fraction`
- `part_reliability_suppression_mean`

Together with the existing route retention, center-only/size-only IoU,
catastrophic-frame, residual concentration, and reliability metrics, these
show whether suppression is selective and whether it reduces tail failures.

## Acceptance gate

V13 is accepted only after epoch-300 full-sequence evaluation. The minimum
comparison is V13 versus V8 and V11 on VisDrone, UAV123, UAVDT, DTB70, and
LaSOT. Targeted paired diagnostics must cover the registered UAV123 and DTB70
sequences in both `ground_truth` and `baseline_replay` modes.

The implementation passing unit and smoke tests is not evidence of an AUC
gain. If V13 does not improve the independent module-one gate, it is rejected;
module-two integration does not waive that requirement.
