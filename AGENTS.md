# Repository Guidelines

## Project Structure & Module Organization

This repository contains PLNet packaged as `hawp`. Core source code lives under `hawp/`: `hawp/fsl/` contains supervised PLNet/HAWP training, model, dataset, config, and benchmark code; `hawp/ssl/` contains self-supervised models, datasets, and training utilities; `hawp/base/` provides shared geometry, visualization, checkpoint, logging, and C++/CUDA extension code. Runtime configs are in `configs/`, docs and figures are in `docs/`, Docker setup is in `docker/`, and evaluation utilities are in `evaluation/`. Keep generated outputs, datasets, and logs out of source directories; use `outputs/` or external folders.

## Build, Test, and Development Commands

- `conda create -n plnet python==3.9 && conda activate plnet`: create the expected environment.
- `pip install -e .`: install the local `hawp` package in editable mode.
- `pip install -r requirement.txt`: install extra runtime dependencies.
- `python -c "import torch; print(torch.cuda.is_available())"`: verify CUDA.
- `python -m hawp.fsl.train configs/plnet.yaml --logdir outputs`: train PLNet on the configured dataset.
- `python -m hawp.fsl.benchmark configs/plnet.yaml --ckpt /path/to/model --dataset wireframe`: evaluate; use `--dataset york` for YorkUrban.
- `docker build -f docker/Dockerfile -t plnet:latest .`: build the Docker image from the repository root.

## Coding Style & Naming Conventions

Use Python 3.9-compatible code and follow the existing style: four-space indentation, snake_case functions and modules, PascalCase classes, and config-driven behavior through YAML/YACS. Keep model, dataset, solver, and utility changes in their existing subpackages. Prefer explicit errors for missing files, checkpoints, CUDA extensions, or datasets.

## Testing Guidelines

No repository-wide test runner or coverage policy is configured. For code changes, run targeted import/compile checks such as `python -m compileall hawp evaluation` and the training, benchmark, or evaluation command affected. Name new tests `test_*.py` and place them near the code under test or in a future top-level `tests/` directory. Avoid large datasets in unit tests; mock paths or use tiny fixtures.

## Commit & Pull Request Guidelines

Recent history uses short imperative summaries such as `update readme` and `add point model`. Keep commits focused and describe the behavior changed. Pull requests should include a concise summary, affected configs or commands, dataset/checkpoint assumptions, and validation results. Include screenshots or rendered figures only when visualization output changes. Do not commit local datasets, generated logs, or large checkpoints unless explicitly required.

## Security & Configuration Tips

Do not hard-code private dataset, checkpoint, or credential paths. Keep machine-specific paths in local configs, command-line arguments, or ignored output directories. Verify downloaded pretrained weights and datasets before reproducible experiments.
