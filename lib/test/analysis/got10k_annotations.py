"""Read-only annotation contract for native GOT-10k identity experiments."""
import numpy as np

from lib.test.analysis.frozen_vdrm_stages import visibility_protocol


def validate_native_annotations(gt, cover, absence, sequence):
    """Keep absent-target sentinels intact; never repair, drop or reorder frames.

    GOT-10k_Train_007643 contains [-1, -1, -1, -1] on absent frames.
    Only that negative-box sentinel with cover=0/absence=1 is accepted;
    initialization still requires a positive box. Other negative geometry
    remains an error, even on an absent frame.
    """
    prefix = f'invalid native annotations: {sequence}: '
    if gt.ndim != 2 or gt.shape[1] != 4 or len(gt) < 2:
        raise ValueError(prefix+'expected at least two GT rows of four values')
    if cover.shape != (len(gt),) or absence.shape != (len(gt),):
        raise ValueError(prefix+'GT/cover/absence lengths or shapes differ')

    def fail_frames(reason, mask):
        frames = (np.flatnonzero(mask)+1).tolist()
        raise ValueError(prefix+f'{reason}; frames (1-based)={frames}')

    finite = np.isfinite(gt).all(1)
    if not finite.all():
        fail_frames('nonfinite GT', ~finite)
    try:
        stages = visibility_protocol(cover, absence)
    except ValueError as error:
        raise ValueError(prefix+str(error)) from error
    if (gt[0, 2:] <= 0).any():
        raise ValueError(prefix+'initial box must have positive width/height; frame (1-based)=1')
    sentinel = (gt == -1).all(1) & (cover == 0) & (absence == 1)
    bad = (gt[:, 2:] < 0).any(1) & ~sentinel
    if bad.any():
        fail_frames('negative box is not an absent [-1,-1,-1,-1] sentinel (cover=0, absence=1)', bad)
    return stages
