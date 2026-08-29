# PET/CT Preprocessing

The configured pipeline is deterministic and runs in this order:

```text
resize_separate -> CT windowing -> paired CT/PET slicing -> CLAHE
```

Run everything with:

```bash
python -m petct preprocess all --config configs/preprocess.yaml
```

Each step is independently runnable by replacing `all` with `resize`, `window`,
`slice`, or `clahe`. The historical scripts in this directory are thin wrappers;
all implementation is in `petct.preprocessing`.

1. `resize_separate.py` reads paired CT, PET, CT-mask, and PET-mask volumes,
   validates image/mask geometry, orients them to RAI, and resamples them. Images
   use linear interpolation and masks use nearest-neighbor interpolation.
2. `cwck.py` applies a CT-only lung window of width 1500 HU and level -600 HU.
   PET intensity is not processed with a CT window.
3. `slice_separate.py` validates the resampled pairs and writes one 64×64×64 CT
   cube and one PET cube. Output case directories use the exact case ID, with no
   `_center` suffix. Files are named `000.png` through `063.png`.
4. `clahe.py` applies configured CLAHE parameters while preserving the cohort,
   modality, and case directory structure.

Every step writes an aggregate JSON summary without case IDs. A failed case is
counted by exception type and is never replaced with fabricated image data.
