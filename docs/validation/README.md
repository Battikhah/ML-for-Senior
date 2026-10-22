# Verification record

Date: 2026-10-18

## Checked locally

- Full suite: **58 passing tests**, with no skipped tests in the tested environment. Tests include all neural architecture gradients, actual fits across all model families, WAV decoding/preparation, connected duplicate/source groups, normalization, cache integrity, teacher/student loss, interrupted-attempt handling, parent checksums, Excel export, SQLite online backup, finalist locks, full INT8 encoding, real float/PTQ/QAT conversions, safe Colab packaging, and private cuDNN setup that preserves existing libraries and avoids repeated downloads. The cuDNN setup test uses filesystem fixtures; it does not load GPU binaries. Conflicting-label regressions use actual WAV files to verify exclusion of every conflicting copy, preservation of source files and group connections, cache reuse, audit evidence, and run/report counts.
- **The delivered `TreeVibes_All_In_One.ipynb` executed top to bottom in a fresh Jupyter kernel**, with all five code cells completing and zero error outputs. This was the explicit CPU/synthetic mode: six training recipes including both PyTorch references, automatic teacher/student training, nine real float/PTQ/QAT conversions, the Excel/HTML/PNG reports, and ESP32 candidate bundles. Artifacts are in `artifacts/one_notebook_verified`; the executed notebook is `artifacts/validation/TreeVibes_All_In_One_smoke_executed.ipynb`. Its synthetic chart was visually inspected. This does not establish Colab GPU execution or a rendered Colab/VS Code UI review.
- Campaign integration tests also exercised optional locked final evaluation and a second invocation with no new training attempts/export jobs. A real-WAV preparation/classical-fit regression verifies that a completed feature cache can resume without calling the raw Kaggle download again.
- Separate Linux x86_64/Python 3.12 dependency sets resolved successfully against PyPI and are shipped under `requirements/`: TensorFlow 2.16.2 uses CUDA runtime 12.3.101/cuDNN 8.9.7.29; PyTorch 2.5.1 uses CUDA runtime 12.4.127/cuDNN 9.1.0.70. Core numerical/training package pins match the local test stack. Native installation and GPU execution of those isolated environments remain pending on Colab.
- Six generated notebooks executed top to bottom in synthetic smoke mode after the recovery/integrity fixes, with zero kernel error outputs. Executed copies are under `artifacts/validation/`; they are intentionally excluded from the source archive.
- Synthetic notebook workflow produced a reference model, quality model, directly trained compact model, teacher and distilled student, plus actual float, INT8 PTQ and INT8 QAT files. These outcomes verify execution paths only.
- Saved comparison chart visually inspected; workbook values read back from all four sheets and selected table projections inspected. This does not establish native Excel rendering or interactive behavior in the Microsoft Excel application.
- Python modules/scripts compiled; all new Python modules, functions and classes have purpose docstrings. All 131 catalog configurations validated.
- Source ZIP contents checked against an explicit source-only allowlist; no datasets, runtime checkpoints, account files or Git metadata included. The VS Code launcher embeds that same ZIP; its unpacking path was exercised in a temporary directory.
- Original notebook SHA-256 remains `d033cf88696d3a940347ed4bf9e64e71acf00fb88b0d0432116cef0e42051e28`.

The tested package versions are in `tested-environment.json`. Upstream warnings remain for Keras/Python deprecations, TFMOT deprecations, TFLite calibration diagnostics, scikit-learn SVM probability deprecation and CPU-core discovery in the sandbox. Actual integer tensor types and interpreter execution were checked despite the converter diagnostic.

The subsequent Python 3.13 bootstrap correction was checked with a real Python 3.13.15 supervisor: it privately installed uv, provisioned Python 3.12.14 and created an isolated worker venv. The updated notebook cells and every embedded Python source compile under Python 3.13. The earlier full notebook execution remains the synthetic workflow baseline; native Colab GPU execution is still pending.

## Independent review

A separate read-only reviewer identified custom Colab output setup, teacher/export artifact integrity, and incomplete attempt initialization/recovery issues. Each received a regression test that failed before its fix and then passed. A Python 3.10 hashing compatibility issue was also fixed and regression-checked. The complete suite was rerun after the fixes. Details and implementation rulings are in `implementation-ledger.md`.

A focused read-only review of the consolidated notebook workflow identified raw-download reuse after runtime replacement. The runner now verifies and reuses its recorded feature caches before fetching raw audio, and lazily fetches pinned Kaggle v1 only when a required feature cache is missing. The raw corpus remains runtime-local to avoid forcing a multi-GB copy onto Drive. The reviewer found no important defect in the revised cache path.

## Still requires external execution

- Complete real TreeVibes dataset/source grouping inspection and field-model training. The user completed the 6.08 GB Kaggle download and reported one byte-identical cross-label pair (two readable WAV files). Preparation now excludes all conflicting copies with an audit trail; its Colab rerun remains pending.
- Actual Colab T4 execution and verification that TensorFlow and PyTorch both use the GPU in the installed runtime. User diagnostics confirm a Tesla T4 with driver 580.82.07 and a working `libcuda.so.1`; TensorFlow 2.16.2 imports but reports missing GPU shared libraries and registers no GPU. Private cuDNN 8 installation and library links are prepared but await a real convolution/backpropagation rerun. A CUDA accelerator selection alone is not framework GPU validation.
- Real-data repeated-seed model selection, compression comparisons and final locked test evaluation.
- ESP-IDF firmware compilation/link size, physical ESP32 inference fixtures, RAM/latency measurements and full application memory headroom.
- Embedded waveform-to-MFCC numeric parity, including capture/filter/FFT/mel/log/DCT differences from the host feature contract.

No field accuracy improvement, T4 speed, or complete ESP32 fit is claimed from synthetic checks or host export alone.
