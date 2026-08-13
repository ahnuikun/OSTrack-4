# VDRM V15 TBR preregistration

## Evidence and scope

V14's targeted paired backend is effectively V8 on the same crop, while its
standard tracker enters different long-lived trajectories on a small set of
sequences. Candidate reliability distinguishes correct from failed frames but
does not distinguish when the VDRM backend is better than the baseline. This
rules out another candidate/reliability gate and does not support another
candidate auxiliary-weight scan.

The existing V8 training crop is already broad. With center jitter 3 and search
factor 4, the target center is sampled over approximately `[0.125, 0.875]` in
normalized crop coordinates. Scale jitter 0.25 already supplies substantial
scale error. Increasing either jitter is therefore not a targeted fix.

V8 full paired diagnostics give the following mean token-level p90 complete
update ratios:

- UAV123 ground truth: 0.3414;
- UAV123 baseline replay: 0.3338;
- DTB70 ground truth: 0.3253;
- DTB70 baseline replay: 0.3244.

## Single intervention

V15 TBR (Tail-Bounded Residual) is V8 `part_aligned` with one parameter-free
trust region on the complete update after bounded LayerScale:

`||delta_i|| <= 0.35 * ||search_token_i||`.

The projection is applied independently per search token only when the raw
update exceeds the bound. It introduces no learned parameters and cannot be
offset by increasing alpha because it is placed after alpha. The input token
norm is detached, so the backbone cannot enlarge the permitted update by
inflating its own norm.

The value 0.35 is fixed before testing. It is just above all four observed V8
mean p90 values, so the intervention targets the observed high-update tail
rather than suppressing the ordinary V8 residual. V4's 0.05 bound was
too restrictive and harmed cross-dataset performance; V15 does not reuse that
setting and no bound scan is permitted.

## Frozen V8 components

- V8 part-aligned route and parameter schema;
- HNCP sampling, probability, placement, scale, and rank objective;
- occlusion augmentation;
- all tracking, visibility, response-rank, and part-route loss weights;
- data sources, center/scale jitter, optimizer, schedule, CE, and head;
- test search/template geometry and tracker feedback behavior.

V15 has no CAIS branch, candidate loss, distillation target, reliability
threshold, temporal state, or module-two connection.

## Required implementation invariants

1. V15 and V8 state-dict keys are exactly equal.
2. For every token below the bound, V15 output is bit-identical to V8 at fixed
   common parameters.
3. Every clipped token update is at most 0.35 of its detached input norm.
4. Zero alpha remains exactly identity.
5. The projection and all trainable V8 parameters have finite gradients.
6. V15 YAML differs from V8 only by `MODEL.VDRM.RESIDUAL_MAX_RATIO`.
7. Diagnostics report clipped-token fraction and mean/min projection scale.

## Mechanism precheck before training

Because the parameter schema is unchanged, the released V8 epoch-300
checkpoint must first be evaluated with the V15 config through the explicit
checkpoint override. This isolates the trust-region mechanism from training
randomness.

Precheck sequences are the known V8-success/V14-failure tails:

- VisDrone `uav0000072_02544_s`;
- UAV123 `uav_uav5`;
- UAVDT `S0103`;
- LaSOT `tank-14`.

TBR is rejected before training if it creates a new loss of 10 AUC or more on
any precheck sequence, or if its mean clip rate is outside `[0.01, 0.20]` on
the targeted paired runs. Passing this precheck only authorizes training; it
does not establish the final module-one Gate.

## Final Gate

Train from the same MAE initialization as V8, never from V8/V14 and never with
a teacher. Run the normal five-suite evaluation and targeted UAV123/DTB70
paired diagnostics in both anchor modes. V15 is accepted only if it avoids new
catastrophic tails and improves the independent module-one Gate. Joint M2
performance cannot waive this requirement.
