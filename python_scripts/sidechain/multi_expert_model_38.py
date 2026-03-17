import os
import json
import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers


# Avoid eager full GPU pre-allocation.
for gpu_device in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu_device, True)
    except Exception:
        pass

print("**************************************************************************************")
print("new model")

FEATURE_DIM = 36
TARGET_DIM = 15
CONT_LEN = 36
RESIDUE_COL = None
BEAD_COL = None

#FEATURE_DIM = 34
#TARGET_DIM = 15
#CONT_LEN = 32
#RESIDUE_COL = 32
#BEAD_COL = 33

class MemmapBatchSequence(tf.keras.utils.Sequence):
    """
    Memory-safe batch loader for large np.memmap datasets.
    Shuffles batch order (not full sample index list) to avoid huge RAM overhead.
    """

    def __init__(self, x, y, m, batch_size, shuffle=True):
        self.x = x
        self.y = y
        self.m = m
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)

        if self.x.shape[0] != self.y.shape[0] or self.x.shape[0] != self.m.shape[0]:
            raise ValueError(
                f"Row mismatch: x={self.x.shape[0]}, y={self.y.shape[0]}, m={self.m.shape[0]}"
            )

        self.n_rows = int(self.x.shape[0])
        self.n_batches = int(math.ceil(self.n_rows / float(self.batch_size)))
        self.batch_order = np.arange(self.n_batches, dtype=np.int32)
        self.on_epoch_end()

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        b = int(self.batch_order[idx])
        start = b * self.batch_size
        end = min(start + self.batch_size, self.n_rows)
        x_batch = np.asarray(self.x[start:end], dtype=np.float32)
        y_batch = np.asarray(self.y[start:end], dtype=np.float32)
        m_batch = np.asarray(self.m[start:end], dtype=np.float32)
        return x_batch, y_batch, m_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.batch_order)



def mlp_res_block(x, units=512, dropout=0.1, kernel_regularizer=None, name="blk"):
    skip = x
    h = layers.LayerNormalization(name=f"{name}_ln0")(x)
    h = layers.Dense(units, kernel_regularizer=kernel_regularizer, name=f"{name}_d0")(h)
    h = layers.LeakyReLU(0.2, name=f"{name}_a0")(h)
    if dropout and dropout > 0:
        h = layers.Dropout(dropout, name=f"{name}_drop0")(h)

    h = layers.Dense(units, kernel_regularizer=kernel_regularizer, name=f"{name}_d1")(h)
    h = layers.LeakyReLU(0.2, name=f"{name}_a1")(h)

    if skip.shape[-1] != units:
        skip = layers.Dense(units, kernel_regularizer=kernel_regularizer, name=f"{name}_proj")(skip)

    out = layers.Add(name=f"{name}_add")([skip, h])
    out = layers.LayerNormalization(name=f"{name}_ln1")(out)
    return out


@tf.keras.utils.register_keras_serializable(package="single_model")
class SliceColumns(layers.Layer):
    def __init__(self, start, end=None, **kwargs):
        super().__init__(**kwargs)
        self.start = int(start)
        self.end = None if end is None else int(end)

    def call(self, x):
        return x[:, self.start:self.end]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"start": self.start, "end": self.end})
        return cfg


@tf.keras.utils.register_keras_serializable(package="single_model")
class RoundClipIndex(layers.Layer):
    def __init__(self, min_index=0, max_index=0, offset=0, **kwargs):
        super().__init__(**kwargs)
        self.min_index = int(min_index)
        self.max_index = int(max_index)
        self.offset = int(offset)

    def call(self, x):
        x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
        x = tf.round(x)
        x = tf.cast(x, tf.int32) - self.offset
        return tf.clip_by_value(x, self.min_index, self.max_index)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "min_index": self.min_index,
                "max_index": self.max_index,
                "offset": self.offset,
            }
        )
        return cfg


def build_single_output_model(
    input_dim=FEATURE_DIM,
    y_dim=TARGET_DIM,
    cont_len=CONT_LEN,
    residue_col=RESIDUE_COL,
    bead_col=BEAD_COL,
    residue_vocab=19,
    residue_emb_dim=8,
    residue_offset=0,
    bead_vocab=4,
    bead_emb_dim=4,
    bead_offset=0,
    trunk_width=512,
    trunk_blocks=6,
    latent_dim=512,
    dropout=0.3,
    l2_reg=1e-5,
):
    kernel_reg = tf.keras.regularizers.l2(float(l2_reg)) if (l2_reg and l2_reg > 0) else None

    x_in = layers.Input(shape=(input_dim,), dtype=tf.float32, name="x")

#    cont = SliceColumns(0, cont_len, name="cont_slice")(x_in)
#    residue_raw = SliceColumns(residue_col, residue_col + 1, name="residue_col")(x_in)
#    bead_raw = SliceColumns(bead_col, bead_col + 1, name="bead_col")(x_in)

#    residue_idx = RoundClipIndex(
#        min_index=0,
#        max_index=residue_vocab - 1,
#        offset=residue_offset,
#        name="residue_idx",
#    )(residue_raw)

#    bead_idx = RoundClipIndex(
#        min_index=0,
#        max_index=bead_vocab - 1,
#        offset=bead_offset,
#        name="bead_idx",
#    )(bead_raw)

#    residue_emb = layers.Embedding(
#        input_dim=residue_vocab,
#        output_dim=residue_emb_dim,
#        name="residue_emb",
#    )(residue_idx)
#    residue_emb = layers.Flatten(name="residue_emb_flat")(residue_emb)

#    bead_emb = layers.Embedding(
#        input_dim=bead_vocab,
#        output_dim=bead_emb_dim,
#        name="bead_emb",
#    )(bead_idx)
#    bead_emb = layers.Flatten(name="bead_emb_flat")(bead_emb)

    # Early fusion so deep trunk can model cont<->categorical interactions.
#    z = layers.Concatenate(name="fuse_in")([cont, residue_emb, bead_emb])

    z = layers.Dense(trunk_width, kernel_regularizer=kernel_reg, name="trunk_d0")(x_in)
    z = layers.LeakyReLU(0.2, name="trunk_a0")(z)
    if dropout and dropout > 0:
        z = layers.Dropout(dropout, name="trunk_drop0")(z)

    for block_index in range(trunk_blocks):
        z = mlp_res_block(
            z,
            units=trunk_width,
            dropout=dropout,
            kernel_regularizer=kernel_reg,
            name=f"trunk_b{block_index}",
        )

    h = layers.Dense(latent_dim, kernel_regularizer=kernel_reg, name="head_d0")(z)
    h = layers.LeakyReLU(0.2, name="head_a0")(h)
    if dropout and dropout > 0:
        h = layers.Dropout(dropout, name="head_drop0")(h)

    y_out = layers.Dense(y_dim, kernel_regularizer=kernel_reg, name="y_out")(h)
    return Model(inputs=x_in, outputs=y_out, name="SingleOutputTabularMLP")


def _coord_num_den(y_true, y_pred, sample_weight, n_atoms=5):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    sample_weight = tf.cast(sample_weight, tf.float32)

    y_true_xyz = tf.reshape(y_true, (-1, n_atoms, 3))
    y_pred_xyz = tf.reshape(y_pred, (-1, n_atoms, 3))
    mask_xyz = tf.reshape(sample_weight, (-1, n_atoms, 3))

    coord_se = tf.square(y_pred_xyz - y_true_xyz)
    num = tf.reduce_sum(coord_se * mask_xyz)
    den = tf.reduce_sum(mask_xyz)
    return num, den


@tf.keras.utils.register_keras_serializable(package="single_model")
class GeometryAwareMaskedMSE(tf.keras.losses.Loss):
    def __init__(self, n_atoms=5, eps=1e-8, name="coord_mse_loss"):
        super().__init__(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name=name)
        self.n_atoms = int(n_atoms)
        self.eps = float(eps)

    def __call__(self, y_true, y_pred, sample_weight=None):
        if sample_weight is None:
            sample_weight = tf.ones_like(y_true, dtype=tf.float32)
        num, den = _coord_num_den(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
            n_atoms=self.n_atoms,
        )
        return num / (den + self.eps)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_atoms": self.n_atoms, "eps": self.eps})
        return cfg


@tf.keras.utils.register_keras_serializable(package="single_model")
class GeometryAwareCoordMSE(tf.keras.metrics.Metric):
    def __init__(self, n_atoms=5, eps=1e-8, name="coord_mse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_atoms = int(n_atoms)
        self.eps = float(eps)
        self.total_num = self.add_weight(name="total_num", initializer="zeros")
        self.total_den = self.add_weight(name="total_den", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is None:
            sample_weight = tf.ones_like(y_true, dtype=tf.float32)
        num, den = _coord_num_den(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
            n_atoms=self.n_atoms,
        )
        self.total_num.assign_add(num)
        self.total_den.assign_add(den)

    def result(self):
        return self.total_num / (self.total_den + self.eps)

    def reset_state(self):
        self.total_num.assign(0.0)
        self.total_den.assign(0.0)


@tf.keras.utils.register_keras_serializable(package="single_model")
class GeometryAwareCoordRMSE(tf.keras.metrics.Metric):
    def __init__(self, n_atoms=5, eps=1e-8, name="rmse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_atoms = int(n_atoms)
        self.eps = float(eps)
        self.total_num = self.add_weight(name="total_num", initializer="zeros")
        self.total_den = self.add_weight(name="total_den", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is None:
            sample_weight = tf.ones_like(y_true, dtype=tf.float32)
        num, den = _coord_num_den(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
            n_atoms=self.n_atoms,
        )
        self.total_num.assign_add(num)
        self.total_den.assign_add(den)

    def result(self):
        return tf.sqrt(self.total_num / (self.total_den + self.eps) + self.eps)

    def reset_state(self):
        self.total_num.assign(0.0)
        self.total_den.assign(0.0)


def main():
    x_train = np.load("local_Frame2/COMBINED_SIDECHAIN/train_features_allPDBs.npy", mmap_mode="r")
    y_train = np.load("local_Frame2/COMBINED_SIDECHAIN/train_targets_allPDBs.npy", mmap_mode="r")
    m_train = np.load("local_Frame2/COMBINED_SIDECHAIN/train_masks_allPDBs.npy", mmap_mode="r")

    x_test = np.load("local_Frame2/COMBINED_SIDECHAIN/test_features_allPDBs.npy", mmap_mode="r")
    y_test = np.load("local_Frame2/COMBINED_SIDECHAIN/test_targets_allPDBs.npy", mmap_mode="r")
    m_test = np.load("local_Frame2/COMBINED_SIDECHAIN/test_masks_allPDBs.npy", mmap_mode="r")


    keep = [i for i in range(38) if i not in (36, 37)]
    x_train = x_train[...,keep]
    x_test = x_test[...,keep]

    if x_train.shape[1] != FEATURE_DIM or x_test.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected feature width {FEATURE_DIM}. Got train={x_train.shape}, test={x_test.shape}"
        )
    if y_train.shape[1] != TARGET_DIM or y_test.shape[1] != TARGET_DIM:
        raise ValueError(
            f"Expected target width {TARGET_DIM}. Got train={y_train.shape}, test={y_test.shape}"
        )
    if m_train.shape[1] != TARGET_DIM or m_test.shape[1] != TARGET_DIM:
        raise ValueError(
            f"Expected mask width {TARGET_DIM}. Got train={m_train.shape}, test={m_test.shape}"
        )

    batch_size = int(os.getenv("BATCH_SIZE", "2048"))
    epochs = int(os.getenv("EPOCHS", "400"))
    run_tag = os.getenv("RUN_TAG", "BASE")
    model_tag = f"SINGLE_{run_tag}"
    model_dir = os.path.abspath(os.getenv("MODEL_DIR", "model_artifacts"))
    os.makedirs(model_dir, exist_ok=True)

    print(f'the new shape of the features is as follows: train: {x_train.shape} the test:  {x_test.shape}')


    init_lr = float(os.getenv("INIT_LR", "1e-4"))
    trunk_blocks = int(os.getenv("TRUNK_BLOCKS", "6"))
    dropout = float(os.getenv("DROPOUT", "0.3"))
    l2_reg = float(os.getenv("L2_REG", "1e-5"))
    residue_vocab = int(os.getenv("RESIDUE_VOCAB", "19"))
    residue_emb_dim = int(os.getenv("RESIDUE_EMB_DIM", "8"))
    residue_offset = int(os.getenv("RESIDUE_OFFSET", "0"))
    bead_vocab = int(os.getenv("BEAD_VOCAB", "4"))
    bead_emb_dim = int(os.getenv("BEAD_EMB_DIM", "4"))
    bead_offset = int(os.getenv("BEAD_OFFSET", "0"))

    early_stop_patience = int(os.getenv("EARLY_STOP_PATIENCE", "25"))
    early_stop_min_delta = float(os.getenv("EARLY_STOP_MIN_DELTA", "1e-5"))
    reduce_lr_patience = int(os.getenv("REDUCE_LR_PATIENCE", "10"))
    min_lr = float(os.getenv("MIN_LR", "1e-6"))

    print("batch_size:", batch_size)
    print("epochs:", epochs)
    print("model_variant: single_output")
    print("model_tag:", model_tag)
    print("model_dir:", model_dir)
    print("residue_col/residue_vocab/residue_emb_dim:", RESIDUE_COL, residue_vocab, residue_emb_dim)
    print("bead_col/bead_vocab/bead_emb_dim:", BEAD_COL, bead_vocab, bead_emb_dim)
    print("train rows (full):", x_train.shape[0])
    print("test rows  (full):", x_test.shape[0])
    print("the shape of X train is:", x_train.shape)
    print("the shape of Y train is as follows:", y_train.shape)
    print("the shape of Mask is as follows:", m_train.shape)


    # Use streaming loaders so TensorFlow does not materialize full datasets as GPU constants.
    train_seq = MemmapBatchSequence(
        x=x_train,
        y=y_train,
        m=m_train,
        batch_size=batch_size,
        shuffle=True,
    )
    val_seq = MemmapBatchSequence(
        x=x_test,
        y=y_test,
        m=m_test,
        batch_size=batch_size,
        shuffle=False,
    )





    strategy = tf.distribute.MirroredStrategy()
    print("Replicas:", strategy.num_replicas_in_sync)

    with strategy.scope():

        model = build_single_output_model(
            trunk_width=512,
            trunk_blocks=trunk_blocks,
            latent_dim=512,
            dropout=dropout,
            l2_reg=l2_reg,
            cont_len=CONT_LEN,
            residue_col=RESIDUE_COL,
            bead_col=BEAD_COL,
            residue_vocab=residue_vocab,
            residue_emb_dim=residue_emb_dim,
            residue_offset=residue_offset,
            bead_vocab=bead_vocab,
            bead_emb_dim=bead_emb_dim,
            bead_offset=bead_offset,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss=GeometryAwareMaskedMSE(n_atoms=5),
            weighted_metrics=[
                GeometryAwareCoordMSE(n_atoms=5, name="coord_mse"),
                GeometryAwareCoordRMSE(n_atoms=5, name="rmse"),
            ],
        )


    model.summary()

    ckpt_all = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join( f"ckpt_epoch_{model_tag}_37" + "_{epoch:02d}.keras"),
        save_weights_only=False,
        save_freq="epoch",
        verbose=1,
    )

    ckpt_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(f"{model_tag}_37_best.weights.h5"),
        monitor="val_coord_mse",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_coord_mse",
        mode="min",
        patience=early_stop_patience,
        min_delta=early_stop_min_delta,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_coord_mse",
        mode="min",
        factor=0.5,
        patience=reduce_lr_patience,
        min_lr=min_lr,
        verbose=1,
    )

    nan_stop = tf.keras.callbacks.TerminateOnNaN()

    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[ckpt_all, ckpt_best,  reduce_lr, nan_stop],
    )

    final_weights_path = os.path.join(model_dir, f"{model_tag}_30_final.weights.h5")
    model.save_weights(final_weights_path)

    # Save a compile-free portable model for inference on other machines/clusters.
    portable_model = tf.keras.models.clone_model(model)
    portable_model.set_weights(model.get_weights())
    portable_model_path = os.path.join(model_dir, f"{model_tag}_30_portable.keras")
    portable_model.save(portable_model_path)

    manifest_path = os.path.join(model_dir, f"{model_tag}_30_manifest.json")
    manifest = {
        "model_tag": model_tag,
        "feature_dim": FEATURE_DIM,
        "target_dim": TARGET_DIM,
        "n_atoms": 5,
        "model_dir": model_dir,
        "best_weights_path": os.path.join(model_dir, f"{model_tag}_30_best.weights.h5"),
        "final_weights_path": final_weights_path,
        "portable_model_path": portable_model_path,
        "hyperparameters": {
            "batch_size": batch_size,
            "epochs": epochs,
            "init_lr": init_lr,
            "trunk_blocks": trunk_blocks,
            "dropout": dropout,
            "l2_reg": l2_reg,
            "cont_len": CONT_LEN,
            "residue_col": RESIDUE_COL,
            "bead_col": BEAD_COL,
            "residue_vocab": residue_vocab,
            "residue_emb_dim": residue_emb_dim,
            "residue_offset": residue_offset,
            "bead_vocab": bead_vocab,
            "bead_emb_dim": bead_emb_dim,
            "bead_offset": bead_offset,
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Saved model artifacts:", manifest_path)

    if "val_coord_mse" in history.history and history.history["val_coord_mse"]:
        best_index = int(np.argmin(history.history["val_coord_mse"]))
        train_mse = float(history.history["coord_mse"][best_index])
        val_mse = float(history.history["val_coord_mse"][best_index])
        train_rmse = float(history.history["rmse"][best_index])
        val_rmse = float(history.history["val_rmse"][best_index])
        print(f"Best epoch by val_coord_mse: {best_index + 1}")
        print(
            f"coord_mse train={train_mse:.6f}, val={val_mse:.6f} | "
            f"rmse train={train_rmse:.6f}, val={val_rmse:.6f}"
        )


if __name__ == "__main__":
    main()
