# RawPLNet Server Training Guide

This guide starts from a fresh server checkout and runs the two intended RawPLNet stages:

1. `adapt_raw`: RGB teacher, synthetic Raw student, point detector/descriptor distillation.
2. `train`: synthetic Raw student, original PLNet line training losses.

The current implementation does not keep an RGB training fallback. RGB is used only as the adapt-stage teacher input.

## 1. Clone And Enter The Repo

```bash
git clone git@github.com:rongjie-yu/RawPLNet.git
cd RawPLNet
```

The original PLNet checkpoint is stored as split parts because the full file is larger than GitHub's normal single-file limit. Reconstruct it after clone:

```bash
python scripts/reconstruct_pretrained.py
```

Confirm the required weights exist:

```bash
ls -lh pretrained/plnet.pth pretrained/invisp_canon.pth
```

Expected files:

- `pretrained/plnet.pth`: original PLNet RGB checkpoint, used to initialize both teacher and student.
- `pretrained/invisp_canon.pth`: InvISP checkpoint used by the RGB to pseudo-Raw synthesis path.

## 2. Prepare The Conda Environment

Use Python 3.9, matching the original PLNet environment expectation.

```bash
conda create -n RawPLnet python=3.9 -y
conda activate RawPLnet

pip install -e .
pip install -r requirement.txt
```

Install the PyTorch build that matches the server CUDA driver. The exact command depends on the server CUDA/runtime stack; verify with the official PyTorch selector before running this step. After installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Do not start training until `torch.cuda.is_available()` prints `True`.

Line training also needs the HAWP C/CUDA extension used by HAFM encoding. The extension is JIT-built by PyTorch and needs CUDA, `nvcc`, a compatible C++ compiler, and `ninja`. Verify it before running the smoke train command:

```bash
mkdir -p outputs/torch_extensions
python -c "from hawp.base.csrc import require_C; print(require_C().encodels)"
```

If this command fails, fix the compiler/CUDA environment first. The line-training smoke command cannot pass while `require_C()` fails.

## 3. Prepare The Dataset

The dataset catalog resolves data under the repository `data/` path. Create a symlink to the prepared Point-Line dataset:

```bash
ln -s /path/to/Point-Line data
```

Required paths for the default config:

```text
data/wireframe/images
data/wireframe/train.json
data/wireframe/test.json
```

Run a quick existence check:

```bash
test -d data/wireframe/images
test -f data/wireframe/train.json
test -f data/wireframe/test.json
```

## 4. Review The Training Config

The default formal training config is:

```text
configs/plnet.yaml
```

Important current settings:

```yaml
DATASETS:
  TRAIN:
  - wireframe_train
  RAW:
    INVISP_CHECKPOINT: pretrained/invisp_canon.pth
    ENABLE_ELD_NOISE: true
    NOISE_MAXSTEP: 15000
    NOISE_RATIO_MIN: 1.0
    NOISE_RATIO_MAX: 50.0
    OUTPUT_MODE: gray_replicated
    DEVICE: cuda
SOLVER:
  IMS_PER_BATCH: 6
  MAX_EPOCH: 40
  BASE_LR: 0.0004
```

`NOISE_MAXSTEP: 15000` means ELD exposure ratio is warmed up over the first 15000 Raw synthesis calls. The current full-noise multiplier range is `[1, 50]`, with:

```text
ratio_dec = min(step, NOISE_MAXSTEP) / NOISE_MAXSTEP
ratio = (uniform(NOISE_RATIO_MIN, NOISE_RATIO_MAX) - 1) * ratio_dec + 1
```

At `ratio_dec=0`, the multiplier is exactly `1`. At `ratio_dec=1`, the multiplier is sampled from `[1, 50]`.

Before a long run, copy the config and edit only machine-specific training settings:

```bash
cp configs/plnet.yaml configs/rawplnet_server.yaml
```

Common server edits:

- `SOLVER.IMS_PER_BATCH`: set according to GPU memory.
- `DATALOADER.NUM_WORKERS`: keep `0` for the current strict global Raw noise warmup behavior. If this is raised above `0`, each DataLoader worker has its own dataset counter, so the Raw noise warmup is no longer a single process-global counter.
- `DATASETS.RAW.NOISE_MAXSTEP`: keep `15000` unless intentionally changing the noise curriculum.
- `DATASETS.RAW.NOISE_RATIO_MIN` and `DATASETS.RAW.NOISE_RATIO_MAX`: keep `1.0` and `50.0` for the current formal full-noise range.
- `DATASETS.RAW.INVISP_CHECKPOINT`: keep `pretrained/invisp_canon.pth` if using the committed checkpoint.

## 5. Preflight Checks

Run static/import checks:

```bash
python -m compileall hawp evaluation
python -m unittest discover tests -v
python -c "from hawp.base.csrc import require_C; print(require_C().encodels)"
```

Run the local debug smoke commands before a long server job:

```bash
python -m hawp.fsl.adapt_raw configs/rawplnet_debug.yaml \
  --teacher-ckpt pretrained/plnet.pth \
  --logdir outputs/debug_adapt \
  --epochs 1 \
  --max-iters 1

python -m hawp.fsl.train configs/rawplnet_debug.yaml \
  --logdir outputs/debug_train \
  --resume outputs/debug_adapt/adapt_raw_final.pth \
  --max-iters 1
```

These commands should complete one adapt iteration and one line-training iteration. They are not quality checks; they only verify the code path, CUDA, checkpoints, data loading, Raw synthesis, and optimizer update path.

## 6. Stage 1: Adapt Raw Student

Run 10 epochs as specified for RawPLNet adaptation:

```bash
python -m hawp.fsl.adapt_raw configs/rawplnet_server.yaml \
  --teacher-ckpt pretrained/plnet.pth \
  --logdir outputs/adapt_raw \
  --epochs 10
```

The default adapt optimizer settings are `--lr 1e-4`, `--lr-scheduler cosine_after_noise`, and `--lr-min 1e-5`. With `cosine_after_noise`, the adapt learning rate stays fixed until `DATASETS.RAW.NOISE_MAXSTEP`, then cosine decays toward `--lr-min` for the remaining adapt steps. `--lr-decay-start` defaults to `DATASETS.RAW.NOISE_MAXSTEP`.

Expected output checkpoints:

```text
outputs/adapt_raw/adapt_raw_noise_warmup_done.pth
outputs/adapt_raw/adapt_raw_epoch_00001.pth
outputs/adapt_raw/adapt_raw_epoch_00002.pth
...
outputs/adapt_raw/adapt_raw_final.pth
```

Adapt data flow:

- Teacher: augmented RGB image -> explicit RGB to grayscale -> original PLNet teacher -> detector and descriptor outputs.
- Student: same augmented RGB image -> InvISP pseudo Raw -> ELD noisy Raw -> demosaiced Raw -> grayscale replicated input -> Raw student.
- Losses: detector cross entropy plus descriptor cosine loss.

The teacher is frozen and in eval mode. The student is initialized from `pretrained/plnet.pth`. Teacher and student use the same augmented/geometrically transformed RGB source image before their branches diverge, so distillation targets remain spatially aligned.

## 7. Stage 2: Formal Line Training

Use the adapt checkpoint as the initialization for the RawPLNet line training stage:

```bash
python -m hawp.fsl.train configs/rawplnet_server.yaml \
  --logdir outputs/train_rawplnet \
  --resume outputs/adapt_raw/adapt_raw_final.pth
```

Formal line training data flow:

```text
RGB image
-> RGB-domain augmentation without gaussian/speckle noise
-> InvISP pseudo Raw
-> ELD noisy Raw
-> demosaiced Raw
-> grayscale replicated input
-> RawPLNet student
-> original PLNet HAFM/LOI/line losses
```

No RGB student training path is used in this stage.

Training outputs are written under:

```text
outputs/train_rawplnet/<config-name>/<timestamp>/
```

Each checkpoint period follows `SOLVER.CHECKPOINT_PERIOD` in the config.

## 8. Resume Or Restart

To resume formal line training from a saved training checkpoint:

```bash
python -m hawp.fsl.train configs/rawplnet_server.yaml \
  --logdir outputs/train_rawplnet \
  --resume /path/to/model_000XX.pth
```

Current `--resume` loads model weights. It does not restore optimizer, scheduler, or epoch counters as a full training-state resume. Treat it as weight initialization unless that behavior is changed and verified.

## 9. Evaluation

After training, evaluate with the PLNet benchmark entry:

```bash
python -m hawp.fsl.benchmark configs/rawplnet_server.yaml \
  --ckpt /path/to/final_checkpoint.pth \
  --dataset wireframe
```

YorkUrban evaluation uses:

```bash
python -m hawp.fsl.benchmark configs/rawplnet_server.yaml \
  --ckpt /path/to/final_checkpoint.pth \
  --dataset york
```

## 10. Server Run Checklist

Before launching a long job, confirm:

- `git status --short` has no unexpected local source changes.
- `python scripts/reconstruct_pretrained.py` has materialized `pretrained/plnet.pth`.
- `data/wireframe/images`, `train.json`, and `test.json` exist.
- `python -m unittest discover tests -v` passes.
- One-iteration adapt and train debug commands complete.
- `configs/rawplnet_server.yaml` uses the intended batch size and keeps `DATASETS.RAW.ENABLE_ELD_NOISE: true`.
- `configs/rawplnet_server.yaml` keeps `DATASETS.RAW.NOISE_MAXSTEP: 15000`, `NOISE_RATIO_MIN: 1.0`, and `NOISE_RATIO_MAX: 50.0` unless you are intentionally running a different noise curriculum.
- The formal line training command uses `--resume outputs/adapt_raw/adapt_raw_final.pth`.
