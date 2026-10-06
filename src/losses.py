"""Loss functions.

Equation 1 of the paper (extremum loss):

    L = MSE(y_hat, y) + alpha * (min_k y_hat_k - min_k y_k)^2
                      + alpha * (max_k y_hat_k - max_k y_k)^2

The window-minimum term (L_min) forces the model to reach the full depth of
southward-Bz storm excursions instead of hedging toward the conditional mean;
the window-maximum term (L_max) acts as a counterweight that prevents a
systematic southward bias. The full model uses alpha = 0.5.

``mse`` and ``minonly`` are provided for the ablation study (Table: ablation).
"""

import tensorflow as tf


def make_loss(name, alpha=0.5):
    if name == "mse":
        def loss(tar, pred):
            return tf.reduce_mean(tf.square(pred - tar))
        return loss

    if name == "minonly":
        def loss(tar, pred):
            mse = tf.reduce_mean(tf.square(pred - tar))
            lmin = tf.reduce_mean(tf.square(tf.reduce_min(pred, -1)
                                            - tf.reduce_min(tar, -1)))
            return mse + alpha * lmin
        return loss

    if name == "minmax":                       # Equation 1 (full model)
        def loss(tar, pred):
            mse = tf.reduce_mean(tf.square(pred - tar))
            lmin = tf.reduce_mean(tf.square(tf.reduce_min(pred, -1)
                                            - tf.reduce_min(tar, -1)))
            lmax = tf.reduce_mean(tf.square(tf.reduce_max(pred, -1)
                                            - tf.reduce_max(tar, -1)))
            return mse + alpha * lmin + alpha * lmax
        return loss

    raise ValueError(f"unknown loss '{name}'")
