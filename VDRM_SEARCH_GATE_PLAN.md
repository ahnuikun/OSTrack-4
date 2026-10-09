# Search gate: cached development screen (2026-10-09)

## Scope frozen before fitting

The verified search-coverage bundle contains 128 previously used guard FIT sequences,
19227 full native non-initial frames and 4848 fixed sampled crop frames. Wide6 fails
the candidate-opportunity gate (20 frames / 6 sequences). Tiles4 has 71 added-good
frames / 10 sequences but naive max-Hann selection causes 23 repairs / 77 damages,
with visible same-anchor AO -1.12693 pp. Keep tiles only as an unproven proposal
mechanism; do not integrate it or revive the failed identity/quality heads.

First answer whether cached causal response/history signals can identify an
unusable native measurement. Fit ONE regularized logistic regression on CPU.
This is a detached-cache diagnostic, not a tracker training arm. It neither changes
any OSTrack checkpoint nor selects a replacement box. There is no GPU collection,
epoch search, architecture sweep, or threshold rescue in this step.

## Inputs and labels

Use all full source-native frames, not only sampled failures. Features take only
native Hann-windowed responses, native predicted boxes and previous predicted
anchors: peak, separated runner-up gap, response entropy/concentration, peak
radius, predicted normalized displacement/log-scale/aspect change, past-only
peak mean/change and motion innovation. Use a trailing 5-frame history. No
de-windowing is presented as an exact raw score, and no current/future GT,
visibility, sequence ID, frame number or crop oracle is an input.

The binary target is `unusable = invisible/invalid OR native IoU < 0.5`.
It is an operational box-usability target, not an identity-error annotation or a
claim to separately measure q_vis/q_id. GT is permitted only for offline labels
and metrics. Inputs are detached saved arrays; reliability gradients cannot reach
the backbone, residual, alpha, or old quality/identity heads.

## Fixed development protocol

- Deterministic seed 42 in code; 5 folds formed by shuffling sequence IDs without
  inspecting labels. For test fold k, fold (k+1)%5 calibrates; the remaining 3
  folds train. No frames from a sequence cross these roles.
- Feature mean/std are fitted on training sequences only. Weight every training
  sequence equally, with no class balancing or nonlinear feature search.
- Fixed L2 coefficient 0.01 on standardized slopes; intercept unpenalized.
  Damped Newton solver, at most 100 iterations, convergence checked. This is not
  a 300-epoch model fine-tune.
- Threshold is fitted ONLY on calibration sequences: flag risk strictly above
  the lowest observed threshold giving at most 10% sequence-macro false positives
  on usable native boxes. Exact ties do not trigger. No test-label quantile tuning.
- Compare against low native peak under exactly the same fold/calibration scheme.
  Report actual out-of-fold FPR and recall (sequence-macro and pooled), 95%
  sequence-bootstrap intervals, Brier and fixed 10-bin ECE. Different achieved
  FPRs are not described as a matched-FPR superiority test.
- A development continuation gate requires >=30 unusable frames / >=10 affected
  sequences, sequence-macro FPR <=10%, and sequence-macro recall >=70%. This is
  a point-estimate screening rule, not statistically confirmed calibration or an
  independent audit pass. Existing 53.5% recall came from a different experiment;
  do not compare it as if sample definitions were identical.

## Decision and artifacts

Pass only authorizes designing conditional crop proposals and a separate
replacement acceptance rule, followed by a NEW train-domain audit excluding all
old head-training, development and guard IDs. It does not approve deployment,
closed-loop tracking, M2, residual effectiveness or Jnew.

Fail stops this response/history-only gate design. Do not adjust its thresholds,
add features or search epochs on these out-of-fold results and call that an audit.
Keep Tclean as the fallback. No new GPU task is issued for a failed gate.

Also report (not optimize) tiles' all-cell versus per-crop-top1 proposal coverage
and near/far naive repairs/damages. This separates candidate availability from
selection competence. Extra-crop arrays occur every four frames and cannot prove
consecutive-frame confirmation or recovery latency.

Output a new, non-overwriting directory containing the source bundle SHA256,
protocol, sequence-role lists, per-fold coefficients/normalization/thresholds,
per-frame labels and out-of-fold scores, and independently reproducible reports.
No production tracker/config files are changed. Existing downloaded files and
weights are left untouched.

## Optional CPU reproduction (not a required server rerun)

```css
python tracking/screen_vdrm_search_gate.py \
  --bundle output/analysis/search_coverage_v1/search_coverage_results.tar.gz \
  --output_dir output/analysis/search_gate_v1
```

There is no seed CLI option. All existing output directories, even empty ones, are rejected.
The local real-data execution and engineering tests will be recorded in the
review document after completion; this plan does not claim results in advance.

Independent analysis reconstructs features, folds, coefficients and thresholds
from the original checked source, then checks every saved prediction and metric:

```css
python tracking/analyze_vdrm_search_gate.py \
  --bundle output/analysis/search_coverage_v1/search_coverage_results.tar.gz \
  --result_dir output/analysis/search_gate_v1
```
