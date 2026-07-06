# Pretrained Weights

This directory stores the checkpoints required by the RawPLNet training flow.

- `invisp_canon.pth`: InvISP checkpoint used by Raw synthesis.
- `plnet.pth.part-00`, `plnet.pth.part-01`: split parts of the original PLNet checkpoint.
- `plnet.pth.sha256`: checksum for the reconstructed `plnet.pth`.

Reconstruct `plnet.pth` before training:

```bash
python scripts/reconstruct_pretrained.py
```

The reconstructed `pretrained/plnet.pth` is ignored by Git because the full file is larger than GitHub's normal single-file limit.
