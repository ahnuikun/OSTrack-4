# VDRM-V12: Positive-Evidence-Preserving Routing

## Scope

V12 is a single-variable extension of the V8/V11 module-one line. It is
trained for 300 epochs from the same MAE initialization as V8 and V11. It does
not use a V8/V11 checkpoint, a teacher, distillation, candidate consensus,
module two, or any additional learned parameter.

The V11 diagnostics showed that background-suppressed routing improved the
same-crop visual backend but reduced the raw residual norm relative to V8.
V12 therefore keeps V11 background suppression and restores only the
aggregate contribution of calibrated positive part routes.

## Formulation

For a part-to-search route logit `l_i`, V8 uses

```text
p_i = sigmoid(l_i)
```

V11 applies a detached monotonic retention factor with residual floor `f`:

```text
r_i = f + (1 - f) * stop_gradient(p_i)
q_i = p_i * r_i
```

V12 uses the existing sigmoid decision boundary; it introduces no threshold
hyperparameter:

```text
M_i = 1[stop_gradient(p_i) >= 0.5]
```

For each sample and template part, the detached preservation scale is

```text
c = sum_i(M_i * p_i) / sum_i(M_i * q_i)  if any M_i = 1
c = 1                                          otherwise
```

and the V12 residual route is

```text
p_i_v12 = c * q_i
```

Part reliability, prototype reconstruction, valid-part normalization, bounded
LayerScale, and all training objectives remain identical to V8/V11.

## Guaranteed invariants

With the configured `f = 0.25`:

- positive route mass is preserved exactly relative to V8;
- `1 <= c <= 2 / (1 + f) = 1.6`, so no arbitrary clipping is required;
- every route with `p_i < 0.5` remains no larger than its V8 contribution;
- if a part has no positive route, its complete route is exactly V11;
- if `f = 1`, the complete forward path is exactly V8;
- zero-initialized LayerScale preserves the original OSTrack forward path;
- the V12 state dictionary is identical to V8/V11:
  `log_match_scale`, `match_bias`, `part_route_log_match_scale`,
  `part_route_match_bias`, and `alpha`.

The mask and scale are detached. Tracking loss can train the original route
probability through `p_i`, but it receives no gradient through positive-set
selection or mass normalization. The existing balanced part-route objective
remains the only direct route calibration loss.

## Ablation chain

| Experiment | Part-aligned route | Background sharpening | Positive mass preservation |
|---|---:|---:|---:|
| Baseline | No | No | No |
| V8 PAR | Yes | No | No |
| V11 BSPAR | Yes | Yes | No |
| V12 PEPR | Yes | Yes | Yes |

No teacher loss, candidate branch, module-two feedback, or additional
parameter changes between V11 and V12. The V11 and V12 YAML files differ only
in `SPATIAL_GATE_MODE`.

## Required checks

Training logs expose:

- `VDRM/part_route_positive_preservation_scale_{mean,min,max}`;
- `VDRM/part_route_positive_mass_ratio` (expected to remain numerically 1);
- `VDRM/part_route_positive_part_fraction`;
- `VDRM/part_route_residual_retention_{mean,min,max}`;
- the existing route probability, residual norm/concentration, and alpha
  diagnostics.

Formal AUC and paired diagnostics are still required. These mathematical and
implementation invariants prevent accidental amplification beyond the V8
background route, but they cannot guarantee a dataset AUC improvement before
the 300-epoch experiment is run.
