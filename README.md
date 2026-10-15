# Multimodal transmission forecasting

Code for forecasting all-sky atmospheric transmission at Cerro Pachón from the DREAM all-sky camera and GOES-East. The forecaster encodes one hour of horizon-locked camera history and the GOES scans available at the forecast time, then decodes each lead (90 s, 5 min, 20 min, 30 min) as a change in natural zenith optical depth. A gate reads the raw satellite sightline now and upwind. Transmission is recovered with the Beer–Lambert airmass factor. The training loss is airmass-weighted yield MSE, with an optional hinge on false clears (observed transmission at or below 0.63).

Camera-only ConvLSTM and SimVP baselines use the same residual and the same loss. Classical baselines are Eulerian persistence, dense Lucas–Kanade optical flow, frozen-flow cross-correlation, and Lagrangian advection by GOES derived-motion winds.

## Layout

- `analysis/forecaster/program.py` — the multimodal model
- `analysis/forecaster/train_multihorizon.py` — training
- `analysis/baselines/neural/` — ConvLSTM and SimVP
- `analysis/baselines/` — classical baselines and holdout scoring
- `analysis/satellite/` — GOES cutouts, derived-motion winds, and the sightline geometry
- `analysis/dataset_pairs/` — packs camera and satellite windows into parquet bundles
- `analysis/our_model/model.py` — encoded pair cache and normalization

## Setup

```bash
pip install -r requirements.txt
export TRANSMISSIONS_MNT=$HOME/transmissions_data
```

`TRANSMISSIONS_MNT` is where caches and checkpoints are written. The default is `~/transmissions_data`. DREAM maps and GOES files are not in this repository. GOES-East is read anonymously from the NOAA public archive:

```bash
python -m analysis.satellite.fetch_pretrain --status
```

Training windows are packed with `python -m analysis.dataset_pairs.pack`. The trainers below build the encoded cache on first use.

Even-index nights are train and validation. Odd-index nights are the holdout. `python -m analysis.forecaster.carve_train_val` writes the night-wise validation index used during training.

## Train

Multimodal forecaster, from scratch:

```bash
python -u -m analysis.forecaster.train_multihorizon --seed 42
```

Two-stage schedule (yield MSE, then the false-clear hinge):

```bash
python -u -m analysis.forecaster.run_curriculum --out runs/seed_42 --seed 42
```

Camera-only baselines:

```bash
python -u -m analysis.baselines.neural.train_multihorizon --arch simvp --seed 42
python -u -m analysis.baselines.neural.train_multihorizon --arch convlstm --seed 42
```

Checkpoints land under `$TRANSMISSIONS_MNT/baselines_neural/`.

## Score

```bash
python -u -m analysis.baselines.score_classical_multihorizon
python -u -m analysis.baselines.score_multihorizon \
    --ckpt $TRANSMISSIONS_MNT/baselines_neural/paper_multih_t24/seed_42/model/best.pt
```

Reported quantities are yield MSE, mean absolute yield error, the false-clear rate at transmission 0.63, the 5σ magnitude change, and the change in effective aperture.
