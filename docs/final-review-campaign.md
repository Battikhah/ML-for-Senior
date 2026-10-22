# Final review, 20 October 2026

The approved experiment menu is implemented separately from the 131-recipe campaign.
Original outputs and scientific modules are read-only. All figures use recorded validation
predictions on exact matching recording IDs; final-test arrays are never loaded.

Execution order: audit/verify the canonical 131 results; restore the two selected CUDA
optimizers and extend to at most 200 epochs (20 new stale checks); extract six alternatives
from exact WAV identities; screen MFCC13, Mel64 and PCEN64 with CNN16; train ImageNet
InceptionResNet-V2 (frozen head then upper blocks, AMP, batch 2 accumulated to 16);
complete remaining feature screens; advance two non-baseline features by recall/F1/FPR.

Continuation preserves optimizer state and current learning rate. A separate new run
would be required to attribute any gain to a lower initial learning rate. Resetting the
stale counter enables a bounded continuation beyond the previous early stop; this is
recorded explicitly. No automatic claim of improvement is made.

Screening uses the same first ten seconds, 8 kHz, 100–2500 Hz filter, 2666-point STFT,
30 time frames, seed, CNN channels, optimizer and budget. Only the representation changes.
Pretrained Inception uses original audio up to 30 seconds, 512-point STFT, 128 mel bands,
128-sample hop, per-recording -80..0 dB scaling and bilinear resizing to 299x299 with
replicated RGB channels and the pretrained [-1,1] scaling. This is paper-inspired.

Output: D:/ML/TreeVibesWindows/final_review_2026-10-20. The final five slides must be
regenerated from completed outputs. RPW is the agreed positive class; healthy is negative.
Board inference remains a candidate until measured on ESP32 hardware.

## Completed evidence

The Windows notebook executed all five code cells. All 131 original model artifact sets
remained intact, and scientific source hashes were unchanged. Thirteen review fits
completed, including the additional direct-PCA16 histogram classifier; the one half-volume
check was a separate fixed-threshold evaluation, not another training campaign.

Full checkpoints remain on the Windows PC. The portable evidence bundle omits raw audio,
feature caches and large PyTorch checkpoints; it contains their recorded identities,
configurations, predictions, tables, plots, and the five-slide PowerPoint/PDF. Slides are
built with the supplied Codex JavaScript presentation runtime and exported to PDF using
its LibreOffice runtime; the notebook regenerates experiments and scientific reports.

The selected 5,048-byte INT8 model and existing ESP32 fixture benchmark are copied into
`best_esp32_candidate/`. This is not a complete acquisition application. The reference
frontend uses filtfilt and a 2666-point FFT, so its memory/streaming implementation must
be validated independently. The current PCA16 C encoder matches eight held-out host
fixtures exactly; neither its latency nor the classifier's RAM/power was measured on ESP32.
