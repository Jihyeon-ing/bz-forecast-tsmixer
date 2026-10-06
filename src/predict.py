"""Run a trained model on the test split (2019-2025) and write results/test_result.npz.

    python src/predict.py                                  # released weights
    python src/predict.py --model_dir trained_model/my_run --out results/my_run.npz

Output .npz keys:
    pred (N, 24), tar (N, 24)      normalised Bz          inp (N, 24, 6)  normalised inputs
    tar_t (N, 24)                  target timestamps      bz_min, bz_max, min_values, max_values, feature_names
Physical Bz [nT] = value * (bz_max - bz_min) + bz_min.  Inference is deterministic.
Evaluate with  python evaluation/evaluate.py --pred results/test_result.npz
"""

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataloader import Dataloader, FEATURE_NAMES     # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="trained_model/tsmixer_bz")
    p.add_argument("--weights", default="best_model.h5")
    p.add_argument("--data", default="data/dataset.npz")
    p.add_argument("--out", default="results/test_result.npz")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    from tensorflow.keras.models import model_from_json

    dl = Dataloader(dataset_path=args.data,
                    scaler_path=os.path.join(args.model_dir, "scaler.json"))
    x_test, y_test, tar_time = dl.get_test()
    print(f"test samples: {x_test.shape}")

    with open(os.path.join(args.model_dir, "model.json")) as f:
        model = model_from_json(f.read())
    model.load_weights(os.path.join(args.model_dir, args.weights))
    pred = model.predict(x_test, batch_size=args.batch, verbose=0)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    s = dl.scaler_dict()
    np.savez_compressed(args.out, pred=pred.astype("float32"), tar=y_test.astype("float32"),
                        inp=x_test, tar_t=tar_time, bz_min=s["bz_min"], bz_max=s["bz_max"],
                        min_values=np.array(s["min_values"]), max_values=np.array(s["max_values"]),
                        feature_names=np.array(FEATURE_NAMES))

    p, t = dl.denorm_bz(pred), dl.denorm_bz(y_test)
    mae = np.abs(p - t).mean(axis=0)
    print("lead(h) :", " ".join(f"{(k + 1) * 0.5:5.1f}" for k in range(mae.size)))
    print("MAE (nT):", " ".join(f"{v:5.2f}" for v in mae))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
