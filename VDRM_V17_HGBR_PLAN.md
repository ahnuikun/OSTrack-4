# VDRM V17 HGBR preregistration

## Evidence and missing link

V3 showed that random same-class hard-negative copy-paste (HNCP) is useful.
V5 aligned the final response-rank negative to the pasted box, but did not
train part routing. V8 then introduced the strongest module-one structure:
each target part is routed and injected using the same calibrated evidence.

V8's route loss nevertheless averages every non-target location into one
background group. A pasted same-class object occupies few retained tokens, so
its loss is diluted by ordinary background. V11's inference-time suppression
showed that route leakage exists, but V11-V13, V15, and V16 also showed that
changing the inference residual can remove useful recovery evidence. V14's
separate identity readout did not directly calibrate the route used by V8.

## Single intervention

V17 HGBR (HNCP Group-Balanced Part Routing) changes only the negative half of
V8's existing part-route loss on samples with a valid pasted HNCP box.

V8 uses:

`L_route = 0.5 * L_target + 0.5 * L_all_background`.

V17 uses, only for successfully applied HNCP samples:

`L_route = 0.5 * L_target`

`        + 0.25 * L_same_class_distractor`

`        + 0.25 * L_ordinary_background`.

Each group is normalized by its own soft negative-target mass. The outer V8
part-route weight remains 0.1. Samples without an applied HNCP box retain the
exact V8 loss and gradient.

The distractor group contains retained CE tokens whose cell centers lie in the
known pasted box and that remain legal negatives for the corresponding target
part. If the box is too small or CE removes every in-box token, the closest
retained legal negative is selected separately for each part. A part with no
legal distractor or ordinary-background token falls back to V8.

## Frozen V8 components

- `part_aligned` inference route, residual, and state-dict schema;
- all model parameters, LayerScale, and residual behavior;
- response-rank, visibility, tracking, and existing part-route objectives;
- every loss weight and auxiliary warm-up;
- HNCP probability, scale, random placement, and source sampling;
- structured occlusion, data, jitter, optimizer, schedule, CE, and head;
- test geometry, tracker feedback, and module-two behavior.

V17 has no new model branch, parameter, inference gate, candidate readout,
threshold, margin, temperature, distillation target, or temporal state. It is
trained from `mae_pretrain_vit_base.pth`, never from a VDRM checkpoint.

## Required invariants

1. V17 and V8 state-dict keys are exactly equal.
2. V17 YAML differs from V8 only by
   `TRAIN.VDRM_GROUP_BALANCE_DISTRACTOR_ROUTE=True`.
3. V17 inference output is bit-identical to V8 for common parameters.
4. When HNCP is absent, V17 route loss and logits gradient are bit-identical
   to V8.
5. Increasing a distractor logit while holding all else fixed strictly
   increases V17 loss.
6. Ordinary-background logits retain finite non-zero gradients.
7. A tiny or CE-removed distractor uses the nearest legal negative; a sample
   with no legal split safely falls back to V8.
8. Part validity and visibility weights remain applied exactly as in V8.
9. Training logs report distractor probability, ordinary-background
   probability, alignment rate, and distractor negative-mass fraction.

## Go/no-go protocol

No inference mechanism precheck is needed because V17's model and forward path
are exactly V8. Before training, unit tests, CUDA smoke, strict V8 checkpoint
loading, and explicit V8/V17 inference equality must pass.

Train one 300-epoch run from the same MAE initialization as V8. Do not scan
group ratios, margins, HNCP probability, placement, loss weight, or warm-up.
After epoch 300, run the five-suite evaluation and registered UAV123/DTB70
paired diagnostics in `ground_truth` and `baseline_replay` modes. V17 is
accepted only if it avoids new catastrophic tails and improves the independent
module-one Gate; module-two performance cannot waive this requirement.
