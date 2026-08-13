# VDRM V16 MCBA preregistration

## Decision basis

V15 showed that residual magnitude is not a valid proxy for harmfulness. A
fixed tail projection improved some sequences but destroyed recovery on
VisDrone `uav0000072_02544_s` and LaSOT `tank-14`. V16 therefore does not
bound residuals, change LayerScale, or introduce a reliability threshold.

V8 already supervises its four route maps against corresponding spatial
subregions of the target. Adding another spatial auxiliary objective would
repeat existing supervision, while V14 showed that an isolated candidate
identity objective does not improve the independent visual Gate. The missing
constraint is in V8's inference route: four independent sigmoids allow several
template parts to claim the same search token without mutual competition.

## Single intervention

V16 MCBA (Mass-Conserving Bidirectional Assignment) replaces only V8's final
part mixture. Let the effective V8 route weight for template part `p` and
search token `j` be:

`w_v8[p,j] = sigmoid(route_logit[p,j]) * part_reliability[p]`.

MCBA computes two existing-evidence probabilities:

1. token-to-part identity probability from `w_v8` across parts;
2. part-to-token spatial probability from `route_logit` across search tokens.

The probabilities are combined in log space and normalized across parts to
obtain `assignment[p,j]`. MCBA then restores the exact V8 scalar route mass at
every token:

`mass_v8[j] = sum_p w_v8[p,j]`

`w_v16[p,j] = mass_v8[j] * assignment[p,j]`

and therefore:

`sum_p w_v16[p,j] = sum_p w_v8[p,j]`.

MCBA changes which target part supplies the feature at a search location. It
does not suppress or amplify the scalar amount of route evidence there. MCBA
adds no learned parameter, temperature, mixing coefficient, top-k, hard
threshold, candidate map, or auxiliary loss.

## Frozen V8 components

- prototype construction and four target-local parts;
- part similarity, global part reliability, and part-route calibration;
- part-route, visibility, rank, and tracking objectives and weights;
- bounded V8 LayerScale, without any new alpha intervention;
- HNCP and structured occlusion augmentation;
- datasets, jitter, optimizer, schedule, CE, head, and test geometry;
- tracker feedback and all module-two behavior.

V16 is trained from `mae_pretrain_vit_base.pth`, never from V8/V14/V15 and
never with a teacher.

## Required implementation invariants

1. V16 and V8 state-dict keys are exactly equal.
2. V16 YAML differs from V8 only in `MODEL.VDRM.SPATIAL_GATE_MODE`.
3. Per-token scalar route mass is conserved to numerical tolerance (`1e-6`).
4. Invalid template parts receive zero assignment.
5. Uniform spatial evidence reduces exactly to the V8 effective part mixture.
6. Zero alpha remains bit-identical to the input.
7. Forward and backward are finite for ordinary, pruned, and partially invalid
   part inputs.
8. Diagnostics report assignment total variation, normalized entropy, and
   maximum mass-conservation error.

## Mechanism precheck before training

The V8 epoch-300 checkpoint must first be evaluated with the V16 config through
the explicit checkpoint override. The unchanged state schema makes this a
direct test of MCBA without training randomness.

Precheck sequences:

- VisDrone `uav0000072_02544_s`;
- UAV123 `uav_uav5`;
- UAVDT `S0103`;
- LaSOT `tank-14`.

MCBA is rejected before training if any sequence loses 10 AUC or more relative
to its published V8 result, produces NaN/Inf, or violates route-mass
conservation above `1e-6`. No temperature, blend, detach, or normalization
variant may be scanned after observing the result.

Passing this precheck authorizes one 300-epoch run from MAE. It does not count
as an independent module-one improvement.

## Final Gate

After epoch 300, run the normal five-suite evaluation and the registered
UAV123/DTB70 targeted paired diagnostics in both `ground_truth` and
`baseline_replay` modes. V16 is accepted only if it avoids new catastrophic
tails and improves the independent module-one Gate. Module-two performance
cannot waive this requirement.
