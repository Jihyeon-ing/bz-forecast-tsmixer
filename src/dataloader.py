"""Data loading, normalisation, split and augmentation (Sections 2 and 3.2).

Dataset file ``data/dataset.npz`` (event-selected OMNI windows, 30-min cadence):
    inp      (N, 24, 6)  12 h of history; features [Bmag, Bz, V, N, P, Ey], Bz = channel 1
    tar      (N, 24)     next 12 h of Bz [nT]
    tar_time (N, 24)     timestamps of the target window

Pipeline
    1. NaNs are linearly interpolated along the time axis (end values extended).
    2. Chronological split by the first target timestamp:
         test  = 2019-2025 (1,335 windows)
         train = 1998-2018; its chronologically LAST 20 % is the validation set.
    3. Min-max normalisation per feature; the target uses the Bz-channel min/max.
         - with ``scaler_path`` the constants are read from a scaler.json
           (use trained_model/tsmixer_bz/scaler.json for the released weights);
         - otherwise they are fitted on the training windows only and should be
           saved next to the new weights (train.py does this).
    4. Jittering augmentation on the TRAINING windows only (after the validation
       split): windows whose normalised target minimum is <= 0.5 (about -6.5 nT)
       are copied ``n_jitter`` times with Gaussian input noise (sigma = 0.02,
       clipped to [0, 1]); targets unchanged. The noise generator is seeded.
"""

import json

import numpy as np
import pandas as pd

BZ_CHANNEL = 1
FEATURE_NAMES = ["Bmag", "Bz", "V", "N", "P", "Ey"]
STORM_THRESHOLD_NORM = 0.5
JITTER_SIGMA = 0.02


def interpolate_time_axis(data):
    """Linear interpolation of NaNs along the time axis, per sample and feature."""
    N, T, F = data.shape
    idx = np.arange(T)
    out = data.astype("float32").copy()
    for n in range(N):
        for f in range(F):
            y = out[n, :, f]
            nan_mask = np.isnan(y)
            if not nan_mask.any():
                continue
            out[n, nan_mask, f] = np.interp(idx[nan_mask], idx[~nan_mask], y[~nan_mask],
                                            left=y[~nan_mask][0], right=y[~nan_mask][-1])
    return out


class Dataloader:
    def __init__(self, dataset_path="data/dataset.npz", scaler_path=None,
                 test_start_year=2019, val_fraction=0.2, n_jitter=9, seed=777):
        self.n_jitter = n_jitter
        self.val_fraction = val_fraction
        self.rng = np.random.default_rng(seed)

        d = np.load(dataset_path, allow_pickle=True)
        inp = interpolate_time_axis(d["inp"])
        tar = interpolate_time_axis(d["tar"][..., None])[..., 0]
        self.tar_time = d["tar_time"]

        years = pd.to_datetime(self.tar_time[:, 0]).year.values
        self.test_index = np.where(years >= test_start_year)[0]
        train_all = np.where(years < test_start_year)[0]          # chronological order
        n_val = int(round(len(train_all) * val_fraction))
        self.train_index = train_all[:-n_val]
        self.val_index = train_all[-n_val:]

        if scaler_path is not None:
            s = json.load(open(scaler_path))
            self.min_values = np.asarray(s["min_values"], dtype=float)
            self.max_values = np.asarray(s["max_values"], dtype=float)
        else:                                   # fit on training windows only
            self.min_values = inp[self.train_index].min(axis=(0, 1)).astype(float)
            self.max_values = inp[self.train_index].max(axis=(0, 1)).astype(float)
        self.bz_min = float(self.min_values[BZ_CHANNEL])
        self.bz_max = float(self.max_values[BZ_CHANNEL])

        self.inp = (inp - self.min_values) / (self.max_values - self.min_values)
        self.tar = (tar - self.bz_min) / (self.bz_max - self.bz_min)

    # ------------------------------------------------------------------ API
    def scaler_dict(self):
        return {"feature_names": FEATURE_NAMES,
                "min_values": self.min_values.tolist(), "max_values": self.max_values.tolist(),
                "bz_min": self.bz_min, "bz_max": self.bz_max}

    def denorm_bz(self, a):
        """Normalised Bz -> physical nT."""
        return np.asarray(a) * (self.bz_max - self.bz_min) + self.bz_min

    def add_jitter(self, x, sigma=JITTER_SIGMA):
        return np.clip(x + self.rng.normal(0.0, sigma, size=x.shape), 0.0, 1.0)

    def get_train(self):
        """Return (x_train, y_train, x_val, y_val); augmentation applied to train only."""
        x, y = self.inp[self.train_index], self.tar[self.train_index]
        storm = y.min(axis=-1) <= STORM_THRESHOLD_NORM
        xs, ys = [x], [y]
        for _ in range(self.n_jitter):
            xs.append(self.add_jitter(x[storm]))
            ys.append(y[storm])
        return (np.concatenate(xs), np.concatenate(ys),
                self.inp[self.val_index], self.tar[self.val_index])

    def get_test(self):
        """Return (x_test, y_test, tar_time) for 2019-2025."""
        i = self.test_index
        return self.inp[i], self.tar[i], self.tar_time[i]
