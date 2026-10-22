# TreeVibes Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Implement directly in the current session; do not delegate unless the user requests it.

**Goal:** Deliver a modular, reproducible Colab experiment suite that compares many infestation classifiers and evaluates compact, distilled, integer-quantized models for ESP32-WROOM-32 deployment.

**Architecture:** Shared modules own data preparation, model creation, training, evaluation, export, and tracking. Named JSON experiment files define independent runs; six Colab notebooks orchestrate selectable stages. Immutable split/feature contracts and saved parent-run identifiers make comparisons traceable.

**Tech Stack:** Python; NumPy/SciPy/librosa; scikit-learn; TensorFlow/Keras for new neural models and integer export; PyTorch for the two reference models; MLflow; pandas/openpyxl; nbformat; pytest; ESP-IDF/TFLite Micro with ESP-NN for the deployment harness.

**Spec:** [Approved design](../specs/2026-10-18-treevibes-experiments-design.md)

## Global Constraints

- Remote inference accepts only a 30 by 13 MFCC matrix; richer input shapes are local-only experiments.
- Use the user's deployment budget of 2 MB flash and approximately 0.5 MB RAM, with no assumed PSRAM.
- These are whole-device limits, not budgets available entirely to model weights or the inference tensor arena.
- Google Colab with a T4 is the primary training environment.
- Do not launch a long training campaign automatically or claim Colab execution without actually running it.
- Preserve the original `TreeVibes_CNN_Training.ipynb`; add new files.
- All functions, classes, and major modules receive concise purpose-oriented docstrings as requested.
- Fit normalization, calibration, and augmentation decisions on training data; select models and thresholds using validation data; keep the test set sealed until finalist selection.
- No model-quality or hardware-fit conclusion is made until the corresponding real experiment or hardware check has run.
- Keep synthetic verification results visibly separate from TreeVibes results.
- Save environment versions and resolved configurations with every run; retain errors and interrupted attempts.

## Review Focus

- Same recording appearing under multiple paths or group identifiers: duplicates must never bridge partitions (Task 1).
- Dataset with too few independent groups or a missing class: fail with an actionable error rather than silently falling back to a leaking split (Task 1).
- Colab disconnect after an epoch or checkpoint write: restore training and MLflow backups without falsely marking the attempt completed (Task 4).
- Teacher checkpoint with a different feature/normalization/split contract: reject distillation before training (Task 3).
- Successful TFLite conversion that emits float tensors or unsupported micro-runtime operators: record an export failure or pending compatibility state, never board verification (Task 5).

## File and interface map

| Files | Responsibility |
| --- | --- |
| `treevibes/config.py` | Read/validate named experiment JSON, resolve defaults, stable configuration fingerprints. |
| `treevibes/data.py` | Manifest, recording/group identities, feature contracts, cache, split and normalization artifacts. |
| `treevibes/models.py` | Keras architecture factories; lazy imports allow data/tracker use without TensorFlow. |
| `treevibes/reference.py` | Faithful PyTorch CNN-A/CNN-B and reference training adapter. |
| `treevibes/train.py` | Supervised/classical/KD training, checkpointing, validation selection. |
| `treevibes/evaluate.py` | Probability metrics, threshold selection, robustness transforms, explicit final evaluation. |
| `treevibes/tracking.py` | Per-attempt files, completion checks, MLflow, workbook and consistent tracking backups. |
| `treevibes/export.py` | INT8 PTQ/QAT conversion, interpreter evaluation, firmware bundle and operator metadata. |
| `scripts/run_experiments.py` | CLI for selected suites, dry runs, bounded execution, resume and final evaluation. |
| `scripts/generate_experiments.py` | Deterministically materialize named experiment JSON files. |
| `scripts/export_leaderboard.py` | Rebuild workbook from persisted run artifacts. |
| `notebooks/01_prepare_data.ipynb` through `06_compare_and_export.ipynb` | Colab entry points using shared modules. |
| `deployment/esp32/` | ESP-IDF inference harness, fixture inputs, generated model adapter and measurement guide. |
| `tests/` | Unit/integration checks grouped by production module, optional-framework checks skipped visibly if unavailable. |
| `pyproject.toml`, `requirements-colab.txt`, `README.md`, `.gitignore` | Installation, runtime setup, reproducible workflow and artifact exclusions. |

Use plain dictionaries for resolved configurations and saved contracts rather than introducing a second configuration framework. `run_dir` and artifact paths use `pathlib.Path`; persisted metadata is JSON. Arrays are NumPy NPZ files loaded with `allow_pickle=False`. Outputs include only serialized numeric/string values, never unsafe object arrays.

### Task 1: Data contracts, extraction and leakage-resistant partitions

**Files:** Create `treevibes/__init__.py`, `treevibes/config.py`, `treevibes/data.py`, `tests/test_data.py`, `tests/test_config.py`, and package/dependency files.

**Interfaces:**
- `load_config(path: Path) -> dict`: validate model family, input shape, inference target, seed, training/compression settings; reject remote shapes other than `[30, 13]`.
- `discover_recordings(root: Path, groups_csv: Path | None = None) -> list[dict]`: path, label, content hash, explicit or discoverable group ID.
- `extract_features(path: Path, feature_config: dict) -> tuple[np.ndarray, dict]`: exact shape plus gate/extraction metadata.
- `make_split(manifest: list[dict], seed: int) -> dict`: persisted row IDs for train/validation/test and grouping evidence.
- `prepare_dataset(root: Path, output: Path, feature_config: dict, groups_csv: Path | None = None) -> Path`: manifest, split, raw feature cache and training-fitted normalization contract.
- `load_dataset(cache: Path) -> dict`: arrays and contracts consumed by later tasks.

- [ ] Write tests asserting 30 by 13 output for short, exact-duration and long WAV fixtures; silence returns a finite SNR and a declared gate result; malformed samples fail visibly.
- [ ] Write split tests asserting no duplicate-content/group overlap, both-class support in every partition, stable row assignment across repeated calls and clear failure for insufficient independent groups. Include a duplicate that connects two otherwise separate groups; its entire connected component must stay together.
- [ ] Write tests that changing one feature setting invalidates the cache; normalization statistics equal the training-only statistics even when validation values are extreme; remote 60 by 20 input is rejected while local input is accepted.
- [ ] Run `python -m pytest tests/test_data.py tests/test_config.py -q`; observe failures for missing behavior before implementation.
- [ ] Implement the interfaces. Match the notebook reference feature calculation with explicit librosa settings. Hash the full contract and recording inventory. Keep original and device-compatible preprocessing variants distinct.
- [ ] Run the same tests and inspect the saved manifest, gate coverage and split metadata on fixtures. Record package versions in the preparation artifact.

### Task 2: Model factories and the broad experiment catalog

**Files:** Create `treevibes/models.py`, `treevibes/reference.py`, `scripts/generate_experiments.py`, JSON files under `experiments/`, and `tests/test_models.py`, `tests/test_catalog.py`.

**Interfaces:**
- `build_model(config: dict)`: new Keras single-logit classifier for the declared input shape.
- `build_reference(config: dict)`: original PyTorch classifier with pooling selected by configuration.
- `generate_catalog(destination: Path) -> list[Path]`: stable, uniquely named JSON files with suite membership.
- All Keras neural families accept channels-last `[frames, coefficients, 1]`; reference adapters convert this at their own boundary. Classical models flatten normalized input internally.

- [ ] Write model tests asserting one finite logit per example for each architecture, valid backward gradients, input shape validation, reference CNN-A/CNN-B pooling differences, and fewer parameters for narrower variants.
- [ ] Write catalog tests asserting stable IDs, no duplicate configurations, valid parent dependencies, multiple model families and both remote/local input targets. Ensure screening/full selections are explicit and no default requests the entire catalog.
- [ ] Run focused tests to observe missing implementation failures.
- [ ] Implement classical families: logistic regression, RBF SVM, random forest and histogram gradient boosting. Keep probabilistic evaluation consistent; record any internal calibration separately.
- [ ] Implement neural families: MLP, temporal CNN, 2D CNN, residual CNN and depthwise separable CNN. Add PyTorch reference models independently without editing the original notebook.
- [ ] Generate at least 60 named experiment configurations across baseline, quality, compact and training-ablation suites, plus explicit compression recipes. Use three finalist seeds `[42, 43, 44]`; screening uses seed 42. Record actual counts in the README generated from the catalog.
- [ ] Include normalization, learning-rate, width, dropout, class-weighting, time-mask/noise augmentation and local feature-size comparisons. Use targeted changes around family baselines rather than multiplying every setting into a full Cartesian grid.
- [ ] Run tests and CLI catalog validation without downloading the real dataset or starting training.

### Task 3: Fair training, KD and evaluation

**Files:** Create `treevibes/train.py`, `treevibes/evaluate.py`, `tests/test_training.py`, `tests/test_evaluate.py`, and `tests/test_distillation.py`.

**Interfaces:**
- `evaluate_probabilities(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict`: accuracy, precision, recall, F1, PR-AUC (average precision), ROC-AUC and fixed `[0,1]` confusion matrix; undefined quantities represented explicitly.
- `select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float`: maximize validation F1 with a deterministic tie-break.
- `train_run(config: dict, dataset: dict, run_dir: Path, teacher_dir: Path | None = None) -> dict`: train and return validation results and checkpoint metadata; no implicit test evaluation.
- `evaluate_final(run_dir: Path, dataset: dict) -> dict`: require a locked finalist manifest before evaluating the untouched test split.
- `distillation_loss(labels, student_logits, teacher_logits, temperature: float, alpha: float)`: hard-label BCE plus Bernoulli soft-target BCE, weighted by alpha and T squared.

- [ ] Write tests for metric values from hand-computable probabilities, all-negative predictions, single-class diagnostics and deterministic threshold ties. Confirm threshold selection accepts validation labels only through the runner path.
- [ ] Write a synthetic supervised-training test checking a real weight update, best-checkpoint restoration and persistent epoch state. Verify train-only augmentation does not modify validation/test arrays.
- [ ] Write KD tests asserting nonzero student gradients, unchanged teacher weights, alpha-zero agreement with hard BCE and rejection of a teacher whose feature, normalization or split fingerprint differs.
- [ ] Run focused tests to observe the expected failures.
- [ ] Implement early stopping, learning-rate scheduling, class weights and seeded training. Seed before constructing each model. Persist optimizer state and epoch state for neural resume. Classical attempts restart explicitly rather than pretending to resume mid-fit.
- [ ] Implement temperature and alpha sweeps for selected teacher/student pairs; each student has a direct-training comparator. Teacher selection references completed validation runs, never test scores.
- [ ] Add deterministic noise/gain/MFCC-precision robustness evaluation with per-perturbation records. Preserve the canonical validation inputs.
- [ ] Run the tests and a tiny synthetic training/KD smoke check. Label every synthetic run with its own dataset identity.

### Task 4: Durable tracking, Colab resume and CLI

**Files:** Create `treevibes/tracking.py`, `scripts/run_experiments.py`, `scripts/export_leaderboard.py`, `tests/test_tracking.py`, and `tests/test_cli.py`.

**Interfaces:**
- `run_experiment(config_path: Path, cache: Path, output: Path, resume: bool = True, teacher_dir: Path | None = None) -> dict`: one isolated attempt with saved failure details.
- `export_leaderboard(output: Path, workbook: Path) -> Path`: rebuild sheets `Runs`, `Experiments`, `Failures`, `Board measurements` from persisted files.
- `backup_tracking(database: Path, destination: Path) -> Path`: consistent SQLite online backup with temporary-file replacement.
- CLI flags include `--suite`, `--config`, `--cache`, `--output`, `--max-runs`, `--dry-run`, `--resume`, and an explicit finalist evaluation command. A dry run prints resolved selections and dependencies without training.

- [ ] Write tests asserting that an interrupted/failed attempt cannot be skipped as completed; changing the config/cache identity creates a new run; missing checkpoints invalidate completion; retries preserve prior failures.
- [ ] Write workbook tests checking exact IDs, numeric metrics, artifact paths and failure rows against saved fixtures. Missing hardware observations remain empty.
- [ ] Write a SQLite backup/restore test that restores committed records into a fresh local database. Add a resume test comparing resumed epoch/checkpoint metadata to the pre-interruption state.
- [ ] Run focused tests to observe failures before implementation.
- [ ] Implement per-attempt directories and atomic result writes. Log parameters, histories, checkpoints and plots to MLflow and per-run JSON. Surface tracking errors in attempt status instead of discarding them.
- [ ] Keep the active SQLite database on the Colab VM; checkpoint artifacts and consistent database backups to the explicitly selected persistent output directory. Document the single-writer assumption.
- [ ] Implement the CLI and workbook export. Default execution selects a small screening preset; full runs require an explicit suite. Handle individual run failures while recording a nonzero final exit status if any selected run fails.
- [ ] Run tests and an end-to-end synthetic preparation/train/resume/report smoke campaign.

### Task 5: Quantization, firmware bundle and evidence levels

**Files:** Create `treevibes/export.py`, `deployment/esp32/CMakeLists.txt`, `deployment/esp32/sdkconfig.defaults`, `deployment/esp32/main/CMakeLists.txt`, `deployment/esp32/main/idf_component.yml`, `deployment/esp32/main/main.cc`, `deployment/esp32/README.md`, `tests/test_export.py`.

**Interfaces:**
- `export_model(run_dir: Path, dataset: dict, method: str, destination: Path) -> dict`: float or full INT8 PTQ/QAT artifact with validation predictions and parent identifiers.
- `encode_input(features: np.ndarray, tensor_details: dict) -> np.ndarray` and `decode_output(values: np.ndarray, tensor_details: dict) -> np.ndarray`: apply declared scale/zero point with clipping and rounding.
- `create_firmware_bundle(export_dir: Path, destination: Path) -> Path`: C model bytes, feature/normalization contract, labels, threshold, required operators and numeric fixtures.
- `import_board_measurement(path: Path, output: Path) -> dict`: validate measurement fields and attach to the exact exported-model hash.

- [ ] Write tests for known int8 encoding/decoding values, clipping boundaries, finite zero-scale rejection, training-only representative calibration and post-conversion validation metrics.
- [ ] Write export tests rejecting float fallback under an INT8 request and preserving conversion failures. Missing board observations must never produce a `board_verified` state.
- [ ] Write a small real conversion test whose integer interpreter predictions are compared with float predictions; skip explicitly when TensorFlow is unavailable, without treating the skip as conversion success.
- [ ] Run focused tests to observe missing implementation failures.
- [ ] Implement PTQ and a small QAT fine-tuning path for compatible selected architectures, using an explicitly tested TensorFlow/Keras/QAT dependency combination. Incompatible models get a clear failed/skipped record and reason; never silently substitute PTQ for QAT.
- [ ] Use actual emitted operators to generate resolver registration and flag any unresolved micro-runtime operation. Fold normalization into export where supported; otherwise emit the firmware transform explicitly.
- [ ] Implement an ESP32-targeted inference harness loading generated reference fixtures and reporting arena usage, available internal heap, largest block, minimum free heap, stack headroom and inference latency. Keep preprocessing latency pending until a numerically verified embedded feature extractor is integrated.
- [ ] Record actual linked image/partition sizes only when an ESP-IDF build is available. Hardware status remains pending until physical execution; an inference-only result cannot certify complete sensing/communications memory fit.
- [ ] Run export tests and host conversion smoke checks where dependencies permit. Document the exact build/flash/measurement commands.

### Task 6: Colab notebooks, replication guide and final verification

**Files:** Create the six notebooks, `README.md`, `.gitignore`, `tests/test_notebooks.py`, and final dependency instructions.

**Interfaces:** Notebooks call the shared package/CLI and store resolved configuration paths, cache identities and output locations in visible setup cells. They do not contain a second training implementation.

- [ ] Write notebook tests asserting valid nbformat structure, parseable Python code cells where applicable, explicit persistent-output setup, no embedded credentials and valid referenced experiment selections.
- [ ] Run the notebook tests to observe missing artifacts.
- [ ] Create notebooks for preparation, references, model quality, compact models, compression/KD, and comparisons/export. Include Colab GPU detection, repository/path setup, dependency installation, selected suite preview, explicit execution cells and resume instructions.
- [ ] Make T4 selection instructions visible. Keep GPU-dependent training distinct from data preparation and report export. Provide a synthetic smoke mode that runs without downloading the 6 GB dataset.
- [ ] Write the replication guide: setup, dataset/group metadata, screening/full stages, teacher selection, three-seed finalists, frozen final test evaluation, tracking restore, Excel export, integer bundle export and hardware evidence boundaries.
- [ ] Document actual catalog counts and commands to list or regenerate configurations. Include experiment-level config examples and explain how to add one architecture without changing the runner.
- [ ] Run `python -m pytest -q`, package/code compilation, configuration validation and feasible synthetic notebooks top to bottom. Inspect the saved notebook/table outputs and workbook. Record any unavailable dependency, dataset, GPU, actual Colab execution or ESP32/toolchain check precisely.
- [ ] Check the original notebook content hash against the pre-implementation hash. Review all new files for accidental data, credentials, checkpoints, local paths and fabricated metric values before delivery.

## Execution and delivery policy

Implementation proceeds inline after plan review. Use the current workspace and preserve the untracked original notebook. No commit, remote push, board flash or full training campaign is required by this plan. The deliverable is the runnable suite plus evidence from feasible checks; real training and board results are reported only after they run.

The user offered Colab access if needed. Request that access when the runnable suite is ready for the real-data T4 campaign, with a concrete notebook and bounded experiment selection. Do not request account credentials in chat.
