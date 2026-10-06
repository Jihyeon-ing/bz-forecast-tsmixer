# Data

All files derive from the NASA/GSFC OMNI database (King & Papitashvili, 2005),
https://omniweb.gsfc.nasa.gov/. Dst and SYM-H are originally provided by the WDC for
Geomagnetism, Kyoto.

## `dataset.npz` — Bz model input/target windows (Section 2)

| key | shape | description |
|---|---|---|
| `inp` | (5409, 24, 6) | 12 h of history, 24 steps x 30 min; features `[Bmag, Bz, V, N, P, Ey]` (Bz = channel 1) |
| `tar` | (5409, 24) | next 12 h of Bz [nT], 24 steps x 30 min |
| `tar_time` | (5409, 24) | timestamps of the target window |

Units: Bmag, Bz [nT]; V [km/s]; N [cm^-3]; P [nPa]; Ey [mV/m].

Construction: 5-min OMNI (1998-2025) averaged to 30 min; sliding windows with 30-min
stride; a window is kept when Bz <= -3 nT at every step of the first 6 h of the target
window; windows with more than half of the input missing are dropped. Remaining NaNs
(0.6 % of target steps, none in the inputs) are linearly interpolated by `src/dataloader.py`.

Split by the first target timestamp: train 1998-2018 (4,074 windows; the chronologically
last 20 % is the validation set), test 2019-2025 (1,335 windows, listed in
`../results/test_sample_times.csv`).

Normalisation: min-max per feature. The constants used by the released weights are in
`../trained_model/tsmixer_bz/scaler.json` (Bz: min -51.355 nT, max 38.43833 nT) and were
fitted on the whole 1998-2025 dataset; the Bz (target) extrema both occur in the training
period. `src/train.py` fits new constants on the training windows only.

## Files for the geomagnetic-index application (Section 4.2)

| file | content | used by |
|---|---|---|
| `omni_hourly_2001_2025.csv` | hourly OMNI: B, Bx, By_GSM, Bz_GSM, T, N, V, Pdyn, Ey, Kp, Dst | O'Brien-McPherron Dst (Figures 4-7) |
| `symh_30min_1995_2025.csv` | 30-min averaged SYM-H | SYM-H horizon extension (Figure 8) |
