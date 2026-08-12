# VDRM V14 CAIS preregistration

## Decision basis

V13 does not show a uniform visual regression. Relative to V8, most sequences
are unchanged or improved, while a small number enter long-lived closed-loop
failure basins. The targeted paired runs also show that scale-only effects are
small and stable, whereas DTB70 loses most of its ground-truth advantage under
baseline replay. V13's part-safety value has essentially random association
with the per-frame VDRM-minus-baseline IoU delta, so another hand-written
reliability threshold is not justified.

V9 is not a suitable candidate-supervision parent. Its candidate focal loss
was coupled to part-route probabilities and its candidate gate multiplied the
visual residual. By epoch 300 the residual energy was highly concentrated and
the learned residual scale saturated. V14 must therefore supervise identity
without either of those forward couplings.

## Single intervention

V14 CAIS (Candidate-Aligned Identity Supervision) uses V8 `part_aligned` as its
only visual parent. It adds one candidate identity readout over the existing
raw cosine part similarities:

1. map each retained part similarity to its original search-grid position;
2. collect local evidence with the fixed radius 1 used by the existing
   candidate diagnostic;
3. average the strongest three valid part responses at each candidate;
4. calibrate the result with one learned scale and bias;
5. supervise the candidate logits with the existing target-centered focal
   objective.

The candidate readout is auxiliary. Its map never multiplies the residual,
never changes the predicted box, and is not connected to module two. The
candidate branch uses raw similarity rather than the calibrated part-route
probability. Its reliability multiplier is detached, so candidate loss cannot
update V8's global reliability scale/bias. It also bypasses the part-route
scale/bias, removing the direct V9 route-collapse mechanism. Candidate loss
can still train template/search features and its own two calibration
parameters, which is the intended identity-discrimination signal.

## Fixed settings

- parent visual path: V8 `part_aligned`;
- insertion layer, HNCP, training data, jitter, residual bound, part-route loss,
  optimizer, schedule, and all test settings: unchanged from V8;
- candidate radius: 1;
- candidate consensus parts: 3;
- candidate calibration initialization: scale 5.0, bias -2.5;
- candidate loss weight: 0.02;
- auxiliary warm-up: 20 epochs.

The weight is fixed before training. V9 used 0.1; near convergence its weighted
candidate objective exceeded the weighted part-route objective. At 0.02 the
same observed loss scale would contribute about one third of the part-route
objective, preserving its role as an auxiliary constraint. No weight scan is
part of this experiment.

## Required invariants before training

1. With common parameters copied, V14's output tokens must equal V8's output
   tokens for nonzero residual scale.
2. Tracking loss alone must not create gradients for the candidate calibration
   parameters.
3. Candidate focal loss must create finite gradients for the candidate
   calibration and input features, but no gradient for global reliability or
   part-route calibration parameters.
4. V14 must expose both candidate and part-route outputs to the actor.
5. The V14 YAML may differ from V8 only by candidate readout settings, spatial
   mode, and candidate loss weight.

## Go/no-go protocol

Train from the same MAE initialization as V8. Do not initialize from V8/V11 or
use any teacher/distillation target. After epoch 300, run the normal five-suite
AUC evaluation and the same targeted UAV123/DTB70 paired set in both
`ground_truth` and `baseline_replay` modes. V14 is a module-one candidate only
if it avoids new catastrophic tails and improves the independent visual gate;
candidate reliability is reported as evidence and does not waive that gate.
