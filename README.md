# bz-forecast-tsmixer: 12-hour forecasting of IMF Bz and its application to geomagnetic indices

Code, trained model, data and reproduction scripts for

> Son, J., Kwak, Y.-S., & Moon, Y.-J. (2026). *Deep Learning-based IMF Bz Forecasting and
> Its Application to Geomagnetic Index Prediction.* Space Weather (submitted).

A TSMixer model forecasts the next 12 hours of the interplanetary magnetic field Bz (GSM,
30-min cadence) from 12 hours of OMNI solar wind history during southward intervals.
Jittering augmentation and an extremum-penalized loss (Eq. 1) improve strong southward
(Bz < -20 nT) predictions. The predicted Bz drives the O'Brien & McPherron (2000) Dst model
and extends the horizon of the deep learning SYM-H model of Son et al. (2026).

[![DOI](https://zenodo.org/badge/1406532534.svg)](https://doi.org/10.5281/zenodo.23176084)

## Layout

```
src/
  model.py             TSMixer architecture (Section 3.1)
  losses.py            extremum-penalized loss, Eq. (1); mse / minonly variants for Table 2
  dataloader.py        interpolation, chronological split, normalisation, jittering (Sections 2, 3.2)
  train.py             training (defaults = released recipe)
  predict.py           deterministic inference on the test split -> results/test_result.npz
  dst_models.py        Burton (1975) and O'Brien & McPherron (2000) Dst equations, Eqs. (3)-(6)
trained_model/tsmixer_bz/
  model.json           architecture          best_model.h5  weights (lowest validation loss)
  scaler.json          normalisation constants tied to the weights
  model.h5             weights at the end of training (kept for completeness)
  SHA256SUMS, training_args_original.json
data/                  dataset.npz (Bz model windows), OMNI hourly and SYM-H for the application
results/
  test_result.npz          released test-set predictions (2019-2025, 1,335 windows)
  test_result_son2025.npz  predictions of the previous model (Son et al., 2025) on the same period
  test_sample_times.csv    start/end time of every test window
  symh_pred_*.npy          SYM-H model outputs used in Figure 8
evaluation/
  evaluate.py          Figure 2, Table 1 and Table 2 numbers from a predictions file
  make_figures.py      Figures 2-8 -> figures/
  ablation/README.md   commands for the five rows of Table 2
notebooks/             exploratory notebooks behind Sections 4.1-4.2
```

## Installation

```bash
conda create -n bz python=3.10
conda activate bz
pip install -r requirements.txt
```

Tested with Python 3.10.15 / TensorFlow 2.10.0 (CPU is sufficient for inference).

## Reproduce the paper's numbers and figures (no training, no TensorFlow needed)

```bash
python evaluation/evaluate.py        # Figure 2 curves, Table 1, Table 2 (full-model row)
python evaluation/make_figures.py    # Figures 2-8 into figures/
```

`evaluate.py` reads `results/test_result.npz` and prints, among others,

| observed Bz range (first 6 h) | N | MAE [nT] | bias [nT] |
|---|---|---|---|
| Bz > -10 | 13,311 | 1.72 | -0.93 |
| -20 <= Bz <= -10 | 2,423 | 3.37 | +1.77 |
| Bz < -20 | 286 | 10.60 | +8.39 |

which is the full-model row of Table 2; Table 1 is computed on the 1,259 windows coincident
with Son et al. (2025).

## Run the released model

```bash
python src/predict.py                # -> results/test_result.npz
```

Loads `trained_model/tsmixer_bz` (model.json + best_model.h5 + scaler.json), runs the 1,335
test windows and writes the predictions file consumed by `evaluation/evaluate.py`.
Inference is deterministic (dropout inactive). Input: 24 x 6 array scaled with
`scaler.json`; output: 24 normalised Bz values (Bz [nT] = value * (bz_max - bz_min) + bz_min).

## Retrain

```bash
python src/train.py --ex_name my_run
```

Defaults reproduce the released recipe: TSMixer (hidden 798, 4 blocks, dropout 0.5),
loss `MSE + 0.5 L_min + 0.5 L_max`, x9 jittering of strong-southward training windows,
Adam lr 1e-3, batch 8, up to 200 epochs, checkpoint with the lowest validation loss,
seed 777. Retrained metrics vary slightly between runs because of GPU non-determinism.
Ablation configurations: `evaluation/ablation/README.md`.

## Citation

See `CITATION.cff`. Please cite both the paper and the Zenodo software record.

## License

MIT (code and model). Data files derived from OMNI are redistributed under the original
NASA/SPDF open-data terms; please acknowledge the OMNI dataset and its PI teams.
