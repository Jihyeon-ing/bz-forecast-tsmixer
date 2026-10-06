# Ablation study (Table 2)

All configurations share the architecture (TSMixer, hidden 798, 4 blocks, dropout 0.5),
the data split and the training hyper-parameters (Adam lr 1e-3, batch 8, up to 200 epochs,
checkpoint with the lowest validation loss, seed 777). Only the augmentation and the loss differ.

Run from the repository root:

```bash
python src/train.py --ex_name abl_mse      --loss mse     --n_jitter 0   # 1. TSMixer + MSE
python src/train.py --ex_name abl_aug      --loss mse     --n_jitter 9   # 2. + augmentation
python src/train.py --ex_name abl_extremum --loss minmax  --n_jitter 0   # 3. + penalized loss (L_min + L_max)
python src/train.py --ex_name abl_minonly  --loss minonly --n_jitter 9   # 4. + aug. + L_min only
python src/train.py --ex_name abl_full     --loss minmax  --n_jitter 9   # 5. full model
```

then, for each run,

```bash
python src/predict.py --model_dir trained_model/abl_mse --out results/abl_mse.npz
python evaluation/evaluate.py --pred results/abl_mse.npz
```

The Table 2 columns are the "Table 2" block of the evaluate.py output (first 6 h; MAE per
range, bias for Bz < -20 nT, and the window-minimum bias).

Note: single-seed runs. Retrained numbers vary slightly between runs because of GPU
non-determinism; the qualitative ordering of the configurations is what Table 2 reports.
