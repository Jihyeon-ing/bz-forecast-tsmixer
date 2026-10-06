"""Train the 12-hour Bz forecast TSMixer (Section 3).

Defaults reproduce the recipe of the released model:

    python src/train.py --ex_name my_run

Ablation configurations (Table 2):
    python src/train.py --ex_name abl_mse      --loss mse     --n_jitter 0
    python src/train.py --ex_name abl_aug      --loss mse     --n_jitter 9
    python src/train.py --ex_name abl_extremum --loss minmax  --n_jitter 0
    python src/train.py --ex_name abl_minonly  --loss minonly --n_jitter 9
    python src/train.py --ex_name abl_full     --loss minmax  --n_jitter 9

Outputs in trained_model/<ex_name>/: model.json (architecture), best_model.h5 (weights of
the epoch with the lowest validation loss), scaler.json (normalisation constants fitted
on the training windows), train_args.json.
"""

import os
import sys
import json
import random
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataloader import Dataloader          # noqa: E402
from model import tsmixer                  # noqa: E402
from losses import make_loss               # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ex_name", required=True)
    p.add_argument("--data", default="data/dataset.npz")
    p.add_argument("--out_root", default="trained_model")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--gpu", default="0")
    # architecture
    p.add_argument("--input_len", type=int, default=24)
    p.add_argument("--tar_len", type=int, default=24)
    p.add_argument("--n_features", type=int, default=6)
    p.add_argument("--hidden_dim", type=int, default=798)
    p.add_argument("--n_blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.5)
    # training
    p.add_argument("--loss", choices=["mse", "minonly", "minmax"], default="minmax")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--n_jitter", type=int, default=9)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--val_fraction", type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    out_dir = os.path.join(args.out_root, args.ex_name)
    os.makedirs(out_dir, exist_ok=True)

    dl = Dataloader(dataset_path=args.data, n_jitter=args.n_jitter,
                    val_fraction=args.val_fraction, seed=args.seed)
    x_tr, y_tr, x_va, y_va = dl.get_train()
    print(f"train {x_tr.shape} (incl. augmentation)  val {x_va.shape}  test {len(dl.test_index)}")

    model = tsmixer(args.input_len, args.n_features, args.tar_len,
                    args.hidden_dim, args.n_blocks, args.dropout)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss=make_loss(args.loss, args.alpha))
    model.summary()

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(out_dir, "best_model.h5"), monitor="val_loss",
        save_best_only=True, save_weights_only=True, verbose=1)
    model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=args.epochs,
              batch_size=args.batch_size, shuffle=True, callbacks=[ckpt], verbose=2)

    with open(os.path.join(out_dir, "model.json"), "w") as f:
        f.write(model.to_json())
    with open(os.path.join(out_dir, "scaler.json"), "w") as f:
        json.dump(dl.scaler_dict(), f, indent=2)
    with open(os.path.join(out_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
