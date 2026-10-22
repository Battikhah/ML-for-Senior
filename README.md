# TreeVibes experiments

This repository now keeps two supported workflows: the Windows RTX GPU campaign and its professor-review notebook. The original 131-recipe campaign and the scientific modules remain available for recovery and audit. The notebooks and reports use `RPW` for the positive class and `healthy` for the negative class.

## Windows GPU campaign

Open [TreeVibes_Windows_GPU_All_In_One.ipynb](TreeVibes_Windows_GPU_All_In_One.ipynb) in Jupyter on the Windows PC and run the cells in order. It downloads and safely extracts the shared experiment ZIP, reuses completed campaign artifacts, and runs remaining neural fits and distillation with PyTorch CUDA on the RTX 5060 Ti. Feature preparation and TensorFlow export/QAT steps still use CPU. The notebook targets `D:/ML/TreeVibesWindows`; keep the PC awake and use one active campaign per output folder.

Rebuild the notebook with:

```bash
python scripts/package_windows_gpu.py
```

## Professor review

[TreeVibes_Final_Professor_Review.ipynb](TreeVibes_Final_Professor_Review.ipynb) packages the separate review workflow. It reuses the Windows GPU and TensorFlow CPU environments, records runs and checkpoints separately from the original campaign, and builds reports from measured outputs. Exported review materials use `artifacts/professor_review_verified_local_2026-10-19/`; that generated output folder and the dated experiment evidence are not present in this checkout.

Rebuild the notebook with:

```bash
python scripts/package_review.py
```

Build the reviewed slides from recorded evidence using the matching `scripts/build_professor_review_verified_local.mjs` and `scripts/finalize_professor_review_verified_local.mjs` tools. Do not substitute planned results for measured runs.

## Data, splits, and recovery

The original campaign uses the TreeVibes recordings and shared 30×13 MFCC contract. Splits isolate duplicate recordings; without tree/session grouping metadata, they do not prove generalization to unseen trees or farms. Threshold selection uses validation data. A held-out test result is reported separately and must not be used to tune the model.

Campaign recovery preserves imported files and reuses verified completed runs and caches. Resume with the same output folder and settings; do not run concurrent writers against one folder. A pre-cleanup source backup was placed at `/tmp/treevibes-before-workflow-cleanup-2026-10-21.zip`; it is a temporary local recovery copy, not durable project storage.

## Deployment limits

The ESP32-WROOM32 target is 2 MB flash and about 0.5 MB RAM. Model-file size alone does not show that the full application fits. The current host-side feature/model checks are not a substitute for measuring flash, tensor arena, latency, and energy on the actual board. PCA16/delta payload results describe feature bytes only; identifiers, timestamps, security, fragmentation, and radio framing add overhead.

## Checks

Install the project and development dependencies in a supported Python 3.10–3.12 environment, then run:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_notebooks.py tests/test_package.py tests/test_final_review.py
```

The focused checks above cover notebook syntax, source packaging, and review contracts. The full suite additionally needs the TensorFlow/PyTorch training dependencies (`.[train,dev]`). Windows GPU execution and ESP32 measurements require their target hardware; local syntax or synthetic checks do not establish those results.
