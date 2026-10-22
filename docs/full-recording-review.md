# Full-recording and gain-normalization review

This isolated follow-up targets the four diagnosed RPW misses. It preserves the original 131-recipe campaign, prior review, accepted recording identities, and final-test holdout.

- Data: 1,709 training recordings and 361 validation recordings. All crops of a recording stay in its original partition. No final-test inputs or labels are loaded.
- Coverage: 10-second windows at 5-second spacing, plus one unique final window that covers the end of each complete WAV. Short recordings receive zero padding.
- Frontend: the existing 8 kHz, 100–2500 Hz bandpass and 30×13 MFCC extraction. Every first-window matrix must match the cached original within 1e-4.
- Controlled comparison: server temporal and local residual architectures, each with either unchanged waveform gain or per-window filtered RMS normalization to 0.01. Both variants start from the same original architecture weights. Each variant fits coefficient scaling on training windows only.
- Training: CUDA, seed 42, fresh Adam at 1e-4, maximum 40 epochs, patience 10, best validation-loss restoration. Recording-level maximum-logit pooling supplies the loss, so longer recordings do not get extra weight and positive clip labels do not label every individual segment positive.
- Evaluation: maximum-logit pooling over the complete recording, validation F1 threshold selection, explicit confusion counts and missed IDs. A separate half-volume evaluation keeps each corresponding clean threshold fixed.
- Deployment: actual INT8 conversion with training-only calibration for the local candidate; train-fitted PCA16 plus real byte encode/decode for the server. PCA16 costs 16 bytes **per window**. The proposed 24-byte local daily summary remains unimplemented firmware, and board RAM/latency/power remain unmeasured.

Replay on the configured Windows server:

```python
import subprocess
subprocess.run([
    r'D:\ML\TreeVibesWindows\runtimes\tf216-cpu\Scripts\python.exe',
    r'D:\ML\TreeVibesWindows\window_review_2026-10-19\window_campaign.py',
    '--previous', r'D:\ML\TreeVibesWindows\final_review_2026-10-20',
    '--output', r'D:\ML\TreeVibesWindows\window_review_2026-10-19',
], check=True)
```

Check `controller.json`, `progress.json`, and the running process before replaying. Do not launch another controller while the first is active. Completed fits are reused and interrupted training resumes from its last checkpoint. Main outputs are `comparison.csv`, `comparison.json`, `conclusions.md`, `runs/`, and `predictions/`.

This is validation-driven model selection, not a fresh independent benchmark or evidence of RPW-specific event recognition. RMS normalization removes scalar amplitude variation, not background noise.

## Finalist audit on 19 October 2026

The reserved test split has now been evaluated **once** for the locked raw PCA16 server finalist, using its previously selected validation F1 threshold (0.2873002513341832). Validation recall was 100%; held-out test recall was 95.83% (92/96 RPW), with 4 RPW misses and 22 healthy false alarms. The locked model, PCA profile, normalization and sources are recorded in `artifacts/pca_audit_2026-10-19/locked_finalist.json`; full results are in `audit_results.json`. Future model selection must not treat this test set as unseen. Earlier reports stating that the test was held out describe their earlier execution state.
