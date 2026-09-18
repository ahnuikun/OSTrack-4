# VDRM V8 A1-A8 Sequential Runbook

Run every command from the repository root. Complete training, checkpoint
validation, four-dataset testing, and analysis for one ablation before starting
the next ablation. Do not launch A1-A8 as one training loop.

All experiments use the same `tracking/train.py` launcher as the earlier VDRM
runs, four visible GPUs, and the same output root. The inner training script's
existing default fixes the base seed at 42. The unique configuration name keeps
checkpoints and results isolated.

## Dataset and environment preflight

`lib/test/evaluation/local.py` must point `settings.visdrone_path` at the
VisDrone root, not its `test` child. The expected layout is:

```text
<visdrone_path>/test/sequences
<visdrone_path>/test/annotations
```

The dataset implementation appends `test` itself. Confirm the resolved first
frame before training:

```bash
python -c "from lib.test.evaluation.environment import env_settings; from lib.test.evaluation import get_dataset; s=env_settings(); d=get_dataset('visdrone'); print('root:', s.visdrone_path); print('sequences:', len(d)); print('first frame:', d[0].frames[0])"
```

The first-frame path must contain `/visdrone/test/sequences/`. Also ensure that
`settings.save_dir` is the same directory passed as `--save_dir` below.

Run the repository tests once before A1:

```bash
python -m unittest discover -s tests -v
```

## A1: V8 structure only

Train:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/train.py \
  --script ostrack \
  --config vitb_256_mae_ce_vdrm_v8_a1_structure_32x4_ep300 \
  --save_dir ./output \
  --mode multiple \
  --nproc_per_node 4 \
  --use_lmdb 0 \
  --use_wandb 0
```

Verify the final checkpoint:

```bash
test -f output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_v8_a1_structure_32x4_ep300/OSTrack_ep0300.pth.tar
```

Test the four datasets:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_v8_a1_structure_32x4_ep300 \
  --dataset all \
  --threads 4 \
  --num_gpus 4
```

Analyze:

```bash
python tracking/analyze_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_v8_a1_structure_32x4_ep300 \
  --dataset all \
  --force_evaluation
```

Stop and review A1 before starting A2.

## A2: A1 plus response-rank loss

Train:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/train.py \
  --script ostrack \
  --config vitb_256_mae_ce_vdrm_v8_a2_rank_32x4_ep300 \
  --save_dir ./output \
  --mode multiple \
  --nproc_per_node 4 \
  --use_lmdb 0 \
  --use_wandb 0
```

Verify the final checkpoint:

```bash
test -f output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_v8_a2_rank_32x4_ep300/OSTrack_ep0300.pth.tar
```

Test the four datasets:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_v8_a2_rank_32x4_ep300 \
  --dataset all \
  --threads 4 \
  --num_gpus 4
```

Analyze:

```bash
python tracking/analyze_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_v8_a2_rank_32x4_ep300 \
  --dataset all \
  --force_evaluation
```

Stop and review A2 before starting A3.

## A3: A1 plus part-route loss

Train:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/train.py \
  --script ostrack \
  --config vitb_256_mae_ce_vdrm_v8_a3_route_32x4_ep300 \
  --save_dir ./output \
  --mode multiple \
  --nproc_per_node 4 \
  --use_lmdb 0 \
  --use_wandb 0
```

Verify the final checkpoint:

```bash
test -f output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_v8_a3_route_32x4_ep300/OSTrack_ep0300.pth.tar
```

Test the four datasets:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_v8_a3_route_32x4_ep300 \
  --dataset all \
  --threads 4 \
  --num_gpus 4
```

Analyze:

```bash
python tracking/analyze_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_v8_a3_route_32x4_ep300 \
  --dataset all \
  --force_evaluation
```

Stop and review A3 before starting A4.

## A4: Rank and part-route losses

Train:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/train.py \
  --script ostrack \
  --config vitb_256_mae_ce_vdrm_v8_a4_rank_route_32x4_ep300 \
  --save_dir ./output \
  --mode multiple \
  --nproc_per_node 4 \
  --use_lmdb 0 \
  --use_wandb 0
```

Verify the final checkpoint:

```bash
test -f output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_v8_a4_rank_route_32x4_ep300/OSTrack_ep0300.pth.tar
```

Test the four datasets:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_v8_a4_rank_route_32x4_ep300 \
  --dataset all \
  --threads 4 \
  --num_gpus 4
```

Analyze:

```bash
python tracking/analyze_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_v8_a4_rank_route_32x4_ep300 \
  --dataset all \
  --force_evaluation
```

Stop and review A4 before starting A5.

## A5: A4 plus structured occlusion

Train:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/train.py \
  --script ostrack \
  --config vitb_256_mae_ce_vdrm_v8_a5_occlusion_32x4_ep300 \
  --save_dir ./output \
  --mode multiple \
  --nproc_per_node 4 \
  --use_lmdb 0 \
  --use_wandb 0
```

Verify the final checkpoint:

```bash
test -f output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_v8_a5_occlusion_32x4_ep300/OSTrack_ep0300.pth.tar
```

Test the four datasets:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_v8_a5_occlusion_32x4_ep300 \
  --dataset all \
  --threads 4 \
  --num_gpus 4
```

Analyze:

```bash
python tracking/analyze_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_v8_a5_occlusion_32x4_ep300 \
  --dataset all \
  --force_evaluation
```

Stop and review A5 before starting A6.

## A6: A5 plus visibility-weighted route supervision

Train:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/train.py \
  --script ostrack \
  --config vitb_256_mae_ce_vdrm_v8_a6_route_visibility_32x4_ep300 \
  --save_dir ./output \
  --mode multiple \
  --nproc_per_node 4 \
  --use_lmdb 0 \
  --use_wandb 0
```

Verify the final checkpoint:

```bash
test -f output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_v8_a6_route_visibility_32x4_ep300/OSTrack_ep0300.pth.tar
```

Test the four datasets:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_v8_a6_route_visibility_32x4_ep300 \
  --dataset all \
  --threads 4 \
  --num_gpus 4
```

Analyze:

```bash
python tracking/analyze_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_v8_a6_route_visibility_32x4_ep300 \
  --dataset all \
  --force_evaluation
```

Stop and review A6 before starting A7.

## A7: A6 plus visibility loss

Train:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/train.py \
  --script ostrack \
  --config vitb_256_mae_ce_vdrm_v8_a7_visibility_loss_32x4_ep300 \
  --save_dir ./output \
  --mode multiple \
  --nproc_per_node 4 \
  --use_lmdb 0 \
  --use_wandb 0
```

Verify the final checkpoint:

```bash
test -f output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_v8_a7_visibility_loss_32x4_ep300/OSTrack_ep0300.pth.tar
```

Test the four datasets:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_v8_a7_visibility_loss_32x4_ep300 \
  --dataset all \
  --threads 4 \
  --num_gpus 4
```

Analyze:

```bash
python tracking/analyze_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_v8_a7_visibility_loss_32x4_ep300 \
  --dataset all \
  --force_evaluation
```

Stop and review A7 before starting A8.

## A8: Full V8 with HNCP

Train:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/train.py \
  --script ostrack \
  --config vitb_256_mae_ce_vdrm_v8_a8_full_32x4_ep300 \
  --save_dir ./output \
  --mode multiple \
  --nproc_per_node 4 \
  --use_lmdb 0 \
  --use_wandb 0
```

Verify the final checkpoint:

```bash
test -f output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_v8_a8_full_32x4_ep300/OSTrack_ep0300.pth.tar
```

Test the four datasets:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_v8_a8_full_32x4_ep300 \
  --dataset all \
  --threads 4 \
  --num_gpus 4
```

Analyze:

```bash
python tracking/analyze_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_v8_a8_full_32x4_ep300 \
  --dataset all \
  --force_evaluation
```

After A8, assemble the four per-dataset AUC values and their unweighted macro
average for A1-A8. Do not average AUC, precision, and normalized precision into
one score.
