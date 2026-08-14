# VDRM V18 ICPR preregistration

## Evidence and scope

V17 is rejected because all five standard datasets regress relative to V8.
Its equal HNCP group weighting promotes a distractor region with roughly four
percent of V8's negative target mass to half of the negative route objective.
V18 does not inherit that intervention.

V8 full paired diagnostics show that failed frames have lower residual norm
and lower top-10 residual-energy concentration than correct frames on UAV123
and DTB70 in both anchor modes. Residual strength is therefore not a reliable
harmfulness signal. V18 may not add a residual bound, alpha restriction,
reliability threshold, candidate gate, or HNCP negative reweighting.

The remaining unmodified source of part identity is V8's prototype pooling:
every template token inside one target quadrant receives equal weight. V18
tests whether reducing internally inconsistent template evidence improves the
localization of the exact V8 route.

## Single intervention

V18 ICPR (Identity-Coherent Part Prototype Refinement) replaces only V8's
uniform template-part pooling. For every valid part it:

1. computes the exact V8 uniform prototype;
2. computes detached cosine agreement between every in-part token and that
   uniform prototype;
3. applies a parameter-free softmax only within the part;
4. detaches the weights and forms a weighted prototype;
5. uses the refined prototype for both V8 route similarity and residual
   injection.

All in-part tokens retain strictly positive weight. There is no temperature,
threshold, top-k, hard deletion, learned selector, auxiliary loss, or new
parameter. Detaching the selector prevents the backbone from manipulating the
weights as a shortcut while preserving gradients through every weighted token
value.

## Frozen V8 components

- `part_aligned` route calibration, reliability, mixture, and residual;
- state-dict schema and V8 LayerScale behavior;
- tracking, visibility, response-rank, and part-route objectives and weights;
- HNCP probability, placement, scale, and ordinary V8 negative treatment;
- structured occlusion, data, jitter, optimizer, schedule, CE, and head;
- test geometry, tracker feedback, and module-two behavior.

The YAML differs from V8 only in `MODEL.VDRM.SPATIAL_GATE_MODE`. V18 starts
from `mae_pretrain_vit_base.pth`, never from a VDRM checkpoint and never with
distillation.

## Required invariants

1. V18 and V8 state-dict keys are exactly equal.
2. One-token parts and uniform-token parts are bit-identical to V8.
3. Valid weights are strictly positive and sum to one; invalid parts stay zero.
4. An inconsistent token receives less weight than coherent tokens.
5. Prototype gradients equal the detached weights and cannot flow through the
   selector.
6. Zero alpha remains bit-identical to the input.
7. Forward and backward are finite for ordinary, tiny, and invalid parts.
8. Logs report normalized weight entropy, effective-token fraction, cosine to
   the V8 uniform prototype, and maximum token weight.

## Mechanism precheck

Strictly load the released V8 epoch-300 checkpoint into V18 and test:

- VisDrone `uav0000072_02544_s`;
- UAV123 `uav_uav5`;
- UAVDT `S0103`;
- DTB70 `Soccer1`;
- LaSOT `tank-14`.

Reject V18 before training if any sequence loses at least 10 AUC, any output
is non-finite, weights fail normalization, or the prototype diagnostics are
absent. No selector, detach, temperature, or blend variant may be scanned.

## Final Gate

After one 300-epoch run from MAE, evaluate the five standard datasets and the
registered UAV123/DTB70 targeted paired sets in `ground_truth` and
`baseline_replay`. V18 must exceed V8's five-suite mean without lowering an
individual dataset, improve at least two datasets, and avoid new catastrophic
tails. Module-two results cannot waive the independent module-one Gate.
