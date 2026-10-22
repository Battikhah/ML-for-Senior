# TreeVibes experiment and ESP32 deployment design

Date: 2026-10-18
Status: draft for review; experiment software and training have not been implemented or run.

## Goal and constraints

Find a strong infestation classifier and determine whether a useful version can run locally on an ESP32-WROOM-32. If inference must run remotely, each recording may transmit only a 30 by 13 MFCC matrix. A local model may use a larger feature matrix if the complete pipeline fits the device.

Use the user's deployment budget of 2 MB flash and approximately 0.5 MB RAM, with no assumed PSRAM. These are whole-device limits, not budgets available entirely to model weights or the inference tensor arena. Do not silently replace them with another module's advertised capacity.

Google Colab with a T4 is the primary training environment. Small-model experiments may also run locally. Prepare runnable experiments rather than requiring the laptop to train the suite. Do not launch a long training campaign automatically or claim Colab execution without actually running it.

The original `TreeVibes_CNN_Training.ipynb` remains the reference. It uses 30 frames in its configuration although several descriptions say 15 frames. New work must describe actual tensor dimensions accurately.

## Approach and alternatives

Recommended: shared Python modules, separate experiment configuration files, focused Colab notebooks, MLflow tracking, and an automatically generated Excel leaderboard. This supplies many inspectable experiments without duplicating training implementations.

An Excel-only tracker is simpler to open but needs custom handling for histories, checkpoints, run identities, and resume state. A separate notebook for every parameter combination duplicates code and makes fair comparisons harder. Neither is the primary implementation.

## Files and responsibilities

```text
notebooks/
  01_prepare_data.ipynb
  02_reference_baselines.ipynb
  03_model_quality.ipynb
  04_compact_models.ipynb
  05_distillation_quantization.ipynb
  06_compare_and_export.ipynb
treevibes/
  data.py              # Discovery, preprocessing, cached features, split manifests.
  models.py            # Quality and compact model definitions.
  train.py             # Supervised training and teacher/student training.
  evaluate.py          # Metrics, validation threshold selection, robustness checks.
  export.py            # Integer export, parity checks, deployment artifact bundle.
  tracking.py          # Durable run records, MLflow logging, Excel export.
experiments/
  baseline/            # Separate named JSON configurations.
  quality/
  compact/
  compression/
scripts/
  run_experiments.py   # Select a suite, resume completed work, limit run count.
  export_leaderboard.py
tests/                # Focused checks of data isolation, KD, export and tracking.
deployment/
  esp32/               # Inference benchmark harness and measurement instructions.
requirements-colab.txt
README.md
```

Actual filenames may be combined when a separate file adds no useful responsibility. All functions, classes, and major modules receive concise purpose-oriented docstrings as requested. Non-obvious edge cases receive short inline comments.

## Data preparation and reproducibility

1. Download the TreeVibes dataset used by the reference notebook, or accept a local dataset root. Validate expected class directories and fail clearly on empty input.
2. Persist a manifest containing recording path, label, content hash, available source/session/tree identifiers, preprocessing outcome, and exclusion reason. Discover grouping metadata from the actual dataset rather than inventing identifiers.
3. Keep duplicates and related recording groups in a single partition where reliable grouping information exists. If the dataset cannot support source-level grouping, label the file-level evaluation limitation explicitly. Do not silently claim unseen-tree or unseen-site generalization.
4. Save one fixed train/validation/test assignment, stratifying classes subject to group isolation. Validate class support and group disjointness. Fail clearly when the requested split is impossible.
5. Fit normalization and any learned preprocessing on training data only. Augmentation and distillation use training records only. Quantization calibration also uses training records only.
6. Cache features by dataset fingerprint and the complete preprocessing configuration. Persist package versions, seed, configuration, split fingerprint, and source revision or source hashes with each run.

Start with the notebook's 8 kHz, 10-second clips, 100–2500 Hz bandpass, SNR gate, and 30 by 13 MFCC features. Treat this as a reproducibility baseline, not proof that host and firmware preprocessing already agree.

Specify every feature setting: frame construction, FFT length, window, mel filters, frequency range, power/log conventions, DCT normalization, padding, and matrix orientation. The original zero-phase filter and host library defaults must not be assumed equivalent to an embedded implementation. A device-compatible feature variant requires its own cache, training runs, and numeric parity fixtures before deployment claims.

Retain gate rejection and extraction error counts by class and partition. Report classifier metrics on accepted inputs together with coverage; discarded recordings must not disappear from the reported population.

## Experiment matrix

Each entry below becomes a named configuration or a small explicit set of configurations. Stages are selectable; opening a notebook must not silently start every experiment.

| Stage | Experiments | Purpose |
| --- | --- | --- |
| Reference | Original PyTorch CNN-A and CNN-B | Establish reproducible reference results at the actual 30 by 13 shape. |
| Input treatment | No normalization versus training-set coefficient normalization; original gate versus ungated diagnostic | Measure preprocessing effects and gate coverage. |
| Quality models | MLP, temporal CNN, wider 2D CNN, residual CNN, depthwise separable CNN | Compare useful architecture families on transmitted 30 by 13 features. |
| Training ablations | Small learning-rate comparison, dropout, class weighting, noise and time masking, early stopping/scheduling | Improve promising families without a combinatorial exhaustive search. |
| Compact models | Small MLP, narrow temporal CNN, narrow 2D CNN, depthwise separable CNN; two widths each | Find the quality versus resource frontier. |
| Distillation | Train each selected student directly and from the selected teacher; small temperature/soft-target-weight grid | Measure the benefit of KD against the same student's supervised baseline. |
| Compression | Float export, representative-data full INT8 PTQ; QAT on selected compatible students | Compare size and post-export quality, including directly trained and distilled students. |
| Local input variants | Selected compact models at 60 by 13 and 60 by 20 | Test whether extra local features justify their preprocessing and activation costs. |
| Robustness | Noise, gain changes, reduced precision of transmitted MFCCs | Quantify sensitivity under declared perturbations. |

Keep all remote-input experiments at 30 by 13. Local input variants are separate experiments and cannot be presented as valid remote fallback models. Transmission precision is not specified by shape alone; log the representation and actual payload bytes in precision experiments.

Use three seeds for finalists, with fixed split assignments. A screening preset limits configurations and epochs; a full preset expands only the explicitly selected stage. No fixed accuracy improvement or elapsed training time is promised.

## Training and distillation

Keep the original PyTorch models as explicit reference implementations. Use TensorFlow/Keras for new deployable architectures and their teachers/students so integer TFLite export does not depend on a fragile cross-framework conversion chain. Cross-framework results are separate runs; architectural similarity is not numeric equivalence.

Use single-logit binary classifiers consistently. The KD loss combines hard-label binary cross entropy with teacher-derived Bernoulli soft targets at temperature T, including the T-squared scaling. Freeze the teacher and prevent gradient updates to it. Teacher selection and hyperparameter selection use validation data. Distillation must load a successfully completed teacher checkpoint with matching feature and split contracts; a missing or incompatible teacher is a visible error.

Select best checkpoints and decision thresholds using validation data. Use validation F1 as the default selection metric, with infested-class recall, precision, PR-AUC, ROC-AUC, accuracy, and confusion counts retained. A later operational minimum-recall requirement can replace the default selection rule without changing the saved predictions.

Do not repeatedly evaluate the test set during model search. Lock finalist configurations, checkpoints, thresholds, and export variants before final test evaluation. Any later search informed by test results must be labeled exploratory.

## Tracking and Colab persistence

Use MLflow for parameters, metric histories, checkpoints, plots, and run comparison. Also persist per-run configuration and result files so the Excel report can be rebuilt without an active tracking server. Export a leaderboard workbook with run results, experiment definitions, failures, and board measurements.

Each run has a unique identifier and explicit state: planned, running, completed, failed, or interrupted. Resume skips only matching completed runs with valid required artifacts. Retries retain earlier failure records. Distillation and export record parent run IDs.

The user explicitly chooses a persistent Colab output directory, normally a mounted Google Drive directory. Record run artifacts there incrementally and save resume checkpoints at epoch boundaries. Keep active MLflow SQLite storage on the Colab VM; create consistent backups to persistent storage and restore them after reconnecting. Do not put a live SQLite database on a mounted Drive filesystem or imply that `/content` survives runtime deletion.

Do not require a publicly exposed tracking server. Notebook tables and the exported workbook provide immediate results; the MLflow UI can be opened locally from the saved tracking bundle. Never claim successful training from placeholder rows.

## ESP32 export and acceptance

Use TensorFlow Lite Micro with Espressif's ESP-NN integration as the initial runtime. Only operators actually emitted by an export can determine its required resolver entries. Export full INT8 models with integer inputs/outputs and training-only representative calibration data. Record conversion failures and unsupported operations rather than quietly permitting float fallback.

Export the model file, C byte array, labels, decision threshold, feature contract, normalization parameters, input/output quantization scales and zero points, required operators, and host reference input/output fixtures. Put normalization in the exported model where practical; otherwise describe exactly how firmware applies it.

For every exported model, evaluate actual interpreter predictions on validation examples. Compare float and integer quality and predictions; use the model's quantization metadata when encoding inputs and decoding outputs. TFLite host execution is not proof of TFLite Micro operator compatibility or hardware fit.

Use these deployment evidence states:

- Exported: converter succeeded and host inference was checked.
- Board candidate: artifact size and operator review justify attempting a build; memory values remain estimates.
- Build verified: the ESP32-targeted firmware links, and actual binary/partition sizes are recorded.
- Board verified: target inference succeeds on reference fixtures, and runtime measurements pass under the intended application configuration.

The benchmark harness records tensor-arena allocation/usage, free internal heap, minimum free heap, largest free block, stack headroom, inference latency, and feature-extraction latency where that extractor is implemented. Account separately for audio capture buffers, MFCC workspace, inference memory, stacks, communications, and firmware. The complete application must fit the actual partition table and runtime memory, with explicit headroom. A successful inference-only harness does not certify the full sensing application.

Treat 2 MB as the total flash ceiling supplied by the user. Read the real partition layout and application requirements before setting the model's available flash budget. Do not equate nominal 0.5 MB RAM with one allocatable block.

No physical board execution is assumed available in this workspace. Deliver the harness and measurement procedure; leave unmeasured hardware fields blank, with a pending state. The unfamiliar names “EPSTA, EPN, and STA” are provisionally interpreted as ESP-DSP, ESP-NN, and ESP-DL, without making them mandatory dependencies.

## Verification and deliverables

- Tests cover exact feature shapes, invalid/silent/short inputs, cache invalidation, split/group isolation, training-only normalization and calibration, model forward shapes, frozen-teacher KD, completed-run resume behavior, failure records, workbook export, and integer input/output encoding.
- Use a tiny synthetic dataset for a training/tracking/export smoke run; label it synthetic and never use its accuracy as evidence about infestation detection.
- Validate notebook schemas, compile code cells where applicable, and run feasible smoke notebooks top to bottom. Declare missing dataset, GPU, dependencies, board/toolchain, or Colab execution explicitly.
- Verify that real final predictions reproduce reported metrics and that workbook rows point to saved run artifacts.
- Preserve the original notebook; add the shared modules, experiment configurations, Colab notebooks, deployment bundle tooling, documentation, and tests as new files.
- No model-quality or hardware-fit conclusion is made until the corresponding real experiment or hardware check has run.

## Primary references checked

- MLflow tracking: https://www.mlflow.org/docs/latest/tracking/
- Espressif TFLite Micro and ESP-NN integration: https://github.com/espressif/esp-tflite-micro
- ESP32 heap instrumentation: https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/system/heap_debug.html
- ESP32-WROOM-32 datasheet: https://documentation.espressif.com/esp32-wroom-32_datasheet_en.html

## Review and next step

Review this design before implementation planning. Default execution scope is preparing and smoke-checking the Colab experiment suite; full dataset training is initiated explicitly in Colab. After design review, write an implementation plan with acceptance checks, then implement the suite in this workspace.
