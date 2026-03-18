# Repository Organization

This repo mixes (1) experiment code, (2) notebooks, and (3) generated artifacts (predictions/plots/tables).
The easiest way to keep it maintainable is to make a clear separation between:

- `src/`: reusable Python package code (importable modules)
- `scripts/`: thin CLI entrypoints that call `src/` code
- `notebooks/`: exploratory work and one-off analysis
- `configs/`: dataset paths, thresholds, model names, and experiment settings
- `artifacts/` (or `runs/`): generated outputs (predictions, logs, model weights); usually gitignored
- `reports/`: paper-ready plots/tables that you want tracked in git

## Suggested Target Tree

```
pig-segmentation-distill/
  README.md
  pyproject.toml
  .env.example
  configs/
    paths.yaml
    teacher_sam3.yaml
    student_yolo.yaml
  src/
    pig_distill/
      __init__.py
      paths.py
      data/
      teacher/
      student/
      eval/
      viz/
  scripts/
    teacher_predict.py
    benchmark_sam3.py
    benchmark_yolo.py
    eval_coco.py
  notebooks/
    dataset_scenarios/
      dataset-distribution-analysis.ipynb
      production_cycle.ipynb
    paper/
      figures.ipynb
  reports/
    figures/
    tables/
  artifacts/
    predictions/
    benchmarks/
    yolo_runs/
```

You don't have to do this in one shot. A safe migration is:

1. Introduce `src/pig_distill/paths.py` that resolves paths relative to the repo root and/or env vars.
2. Update scripts to stop using absolute paths like `/hd2/...` and instead accept `--data-root`, `--output-dir`.
3. Move notebooks into `notebooks/` (keep their outputs under `reports/` if you want them versioned).
4. Decide what is tracked in git:
   - Track: `reports/` (paper-ready outputs), small `results/*.json`, tables.
   - Ignore: `artifacts/` (large, frequently regenerated outputs), local datasets under `data/`.

## Conventions That Pay Off Quickly

- One place for paths: no hard-coded absolute paths inside modules. Prefer:
  - env vars: `DATA_ROOT`, `ARTIFACTS_DIR`
  - CLI args: `--data-root`, `--out`
  - config files under `configs/`
- One place for entrypoints: `scripts/` should be runnable and short; business logic lives in `src/`.
- Make outputs deterministic:
  - `artifacts/<run_id>/predictions.json`
  - `artifacts/<run_id>/metrics.json`
  - `reports/figures/<figure_name>.pdf`
- Keep notebooks importable:
  - prefer `from pig_distill... import ...` over copying code into notebooks.

## Mapping From Current Repo

Current folders map cleanly to the target:

- `teacher/` -> `src/pig_distill/teacher/` (model loading + prediction generation) and `scripts/teacher_predict.py`
- `student/` -> `src/pig_distill/student/` (training/inference utilities) and `scripts/...`
- `benchmark/` -> `scripts/benchmark_*.py` + `src/pig_distill/benchmark/` if you want shared logic
- `results/scripts/` -> `src/pig_distill/eval/` and `scripts/eval_*.py`
- `dataset-scenarios/` and `paper/` -> `notebooks/` (or keep as-is, but separate them from source code)

If you want, I can implement the first "safe migration" step (central path handling + CLI args) without moving files yet.

