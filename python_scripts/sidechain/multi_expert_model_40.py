import os
import json
import math
import importlib.util
from pathlib import Path

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

FEATURE_DIM = 38
TARGET_DIM = 15
CONT_LEN = 36
RESIDUE_COL = 36
BEAD_COL = 37
N_ATOMS = TARGET_DIM // 3
DEFAULT_BOND_SCALE = 7.0
DEFAULT_BOND_EPS = 1e-8


def _env_flag(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_bond_lookup2():
    """
    Load bond topology tables from bond_lookup2.py regardless of launch path.
    """
    this_file = Path(__file__).resolve()
    candidates = [
        this_file.parent / "bond_lookup.py",
        this_file.parent.parent / "bond_lookup.py",
        this_file.parent.parent.parent / "bash_scripts" / "bond_lookup.py",
        Path.cwd() / "bond_lookup.py",
        Path.cwd() / "bash_scripts" / "bond_lookup.py",
    ]

    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("bond_lookup", str(candidate))
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.INT_TO_AA, module.ATOM_ORDER, module.ATOM_GRAPH, str(candidate)

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "bond_lookup.py not found. Searched:\n" + searched
    )


BOND_INT_TO_AA, BOND_ATOM_ORDER, BOND_ATOM_GRAPH, BOND_LOOKUP2_PATH = _load_bond_lookup2()
NUM_RES_TYPES = max(int(k) for k in BOND_INT_TO_AA.keys()) + 1
MAX_BEADS = max(max(bead_map.keys()) for bead_map in BOND_ATOM_ORDER.values()) + 1

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


class BondWeightScheduler(tf.keras.callbacks.Callback):
    """
    Coordinate-first schedule:
      1) coord-only phase (bond weight = start_weight, usually 0.0)
      2) linear ramp to target weight
      3) hold target weight
    """

    def __init__(
        self,
        target_weight,
        coord_only_epochs=20,
        ramp_epochs=40,
        start_weight=0.0,
        verbose=1,
    ):
        super().__init__()
        self.target_weight = float(target_weight)
        self.coord_only_epochs = int(coord_only_epochs)
        self.ramp_epochs = int(ramp_epochs)
        self.start_weight = float(start_weight)
        self.verbose = int(verbose)
        self._last_weight = None

    def _weight_for_epoch(self, epoch_idx):
        if epoch_idx < self.coord_only_epochs:
            return self.start_weight

        if self.ramp_epochs <= 0:
            return self.target_weight

        ramp_pos = epoch_idx - self.coord_only_epochs + 1
        t = min(max(float(ramp_pos) / float(self.ramp_epochs), 0.0), 1.0)
        return (1.0 - t) * self.start_weight + t * self.target_weight

    def on_train_begin(self, logs=None):
        if not hasattr(self.model, "bond_weight"):
            raise AttributeError("Model must expose `bond_weight` tf.Variable for BondWeightScheduler.")
        w = self._weight_for_epoch(0)
        self.model.bond_weight.assign(w)
        self._last_weight = float(w)
        if self.verbose:
            print(f"[BondWeightScheduler] epoch=1 bond_weight={w:.6f}")

    def on_epoch_begin(self, epoch, logs=None):
        w = self._weight_for_epoch(int(epoch))
        self.model.bond_weight.assign(w)
        if self.verbose and (self._last_weight is None or abs(w - self._last_weight) > 1e-12):
            print(f"[BondWeightScheduler] epoch={epoch + 1} bond_weight={w:.6f}")
        self._last_weight = float(w)

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs["bond_weight"] = float(self.model.bond_weight.numpy())


def build_bond_tables(int_to_aa, atom_order, atom_graph, num_res=None, max_beads=None):
    """
    Build compact tensor lookup tables for bond pairs per (residue_id, bead_id).

    Returns:
      bond_pairs_table: (T, K, 2) int32 with local-atom pair indices, padded by -1
      bond_valid_table: (T, K) float32 mask for valid bond rows
      template_of_res_bead: (num_res, max_beads) int32 template id per route, -1 if none
      table_stats: dict with lightweight summary statistics
    """
    if num_res is None:
        num_res = max(int(k) for k in int_to_aa.keys()) + 1
    if max_beads is None:
        max_beads = max(max(bead_map.keys()) for bead_map in atom_order.values()) + 1

    templates = []
    template_of_res_bead = -np.ones((num_res, max_beads), dtype=np.int32)

    for res_id in range(num_res):
        if res_id not in int_to_aa:
            continue
        res_name = int_to_aa[res_id]
        if res_name not in atom_order or res_name not in atom_graph:
            continue

        for bead_id in range(max_beads):
            if bead_id not in atom_order[res_name] or bead_id not in atom_graph[res_name]:
                continue

            atom_list = atom_order[res_name][bead_id]
            atom_to_idx = {name: idx for idx, name in enumerate(atom_list)}

            bonds = set()
            for atom_name, neighbors in atom_graph[res_name][bead_id]:
                if atom_name not in atom_to_idx:
                    continue
                atom_idx = atom_to_idx[atom_name]
                for neighbor_name in neighbors:
                    if neighbor_name not in atom_to_idx:
                        continue
                    neighbor_idx = atom_to_idx[neighbor_name]
                    if atom_idx == neighbor_idx:
                        continue
                    i, j = (atom_idx, neighbor_idx) if atom_idx < neighbor_idx else (neighbor_idx, atom_idx)
                    bonds.add((i, j))

            bond_pairs = sorted(bonds)
            template_id = len(templates)
            templates.append(bond_pairs)
            template_of_res_bead[res_id, bead_id] = template_id

    max_bonds = max((len(t) for t in templates), default=1)
    num_templates = len(templates)
    bond_pairs_table = -np.ones((num_templates, max_bonds, 2), dtype=np.int32)
    bond_valid_table = np.zeros((num_templates, max_bonds), dtype=np.float32)

    for template_id, bond_pairs in enumerate(templates):
        for bond_idx, (i_idx, j_idx) in enumerate(bond_pairs):
            bond_pairs_table[template_id, bond_idx, 0] = i_idx
            bond_pairs_table[template_id, bond_idx, 1] = j_idx
            bond_valid_table[template_id, bond_idx] = 1.0

    table_stats = {
        "num_templates": int(num_templates),
        "max_bonds_per_template": int(max_bonds),
        "routed_slots": int(np.count_nonzero(template_of_res_bead >= 0)),
        "total_bond_targets": int(np.count_nonzero(bond_valid_table > 0.0)),
    }

    return (
        tf.constant(bond_pairs_table, dtype=tf.int32),
        tf.constant(bond_valid_table, dtype=tf.float32),
        tf.constant(template_of_res_bead, dtype=tf.int32),
        table_stats,
    )


def bond_mse_from_tables(
    x,
    y_true,
    y_pred,
    mask,
    bond_pairs_table,
    bond_valid_table,
    template_of_res_bead,
    n_atoms=5,
    residue_col=36,
    residue_offset=0,
    num_res_types=19,
    bead_col=37,
    max_beads=4,
    scale=7.0,
    eps=1e-8,
):
    """
    Bond-only loss:
      - route each sample by (residue_id, bead_id)
      - gather local bond pairs from bond_lookup2 topology
      - compare predicted vs true bond lengths in Angstrom
    """
    x = tf.cast(x, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask = tf.cast(mask, tf.float32)

    residue_raw = x[:, residue_col]
    residue_raw = tf.where(tf.math.is_finite(residue_raw), residue_raw, tf.zeros_like(residue_raw))
    residue_id = tf.cast(tf.round(residue_raw), tf.int32) - int(residue_offset)
    residue_id = tf.clip_by_value(residue_id, 0, int(num_res_types) - 1)

    bead_raw = x[:, bead_col]
    bead_raw = tf.where(tf.math.is_finite(bead_raw), bead_raw, tf.zeros_like(bead_raw))
    bead_id = tf.cast(tf.round(bead_raw), tf.int32)
    bead_id = tf.clip_by_value(bead_id, 0, int(max_beads) - 1)

    route_idx = tf.stack([residue_id, bead_id], axis=1)  # (B, 2)
    template_id = tf.gather_nd(template_of_res_bead, route_idx)  # (B,)
    template_valid = tf.cast(template_id >= 0, tf.float32)

    template_safe = tf.maximum(template_id, 0)
    pairs = tf.gather(bond_pairs_table, template_safe)      # (B, K, 2)
    pair_valid = tf.gather(bond_valid_table, template_safe) # (B, K)

    true_xyz = tf.reshape(y_true, (-1, n_atoms, 3))
    pred_xyz = tf.reshape(y_pred, (-1, n_atoms, 3))
    mask_xyz = tf.reshape(mask, (-1, n_atoms, 3))

    atom_present = tf.cast(tf.reduce_any(mask_xyz > 0.0, axis=-1), tf.float32)  # (B, n_atoms)

    i_idx = pairs[:, :, 0]
    j_idx = pairs[:, :, 1]
    i_safe = tf.maximum(i_idx, 0)
    j_safe = tf.maximum(j_idx, 0)

    true_i = tf.gather(true_xyz, i_safe, batch_dims=1)
    true_j = tf.gather(true_xyz, j_safe, batch_dims=1)
    pred_i = tf.gather(pred_xyz, i_safe, batch_dims=1)
    pred_j = tf.gather(pred_xyz, j_safe, batch_dims=1)

    d_true = tf.sqrt(tf.reduce_sum(tf.square(true_i - true_j), axis=-1) + eps) * scale
    d_pred = tf.sqrt(tf.reduce_sum(tf.square(pred_i - pred_j), axis=-1) + eps) * scale

    atom_i_present = tf.gather(atom_present, i_safe, batch_dims=1)
    atom_j_present = tf.gather(atom_present, j_safe, batch_dims=1)
    not_padded = tf.cast((i_idx >= 0) & (j_idx >= 0), tf.float32)

    valid = (
        pair_valid
        * atom_i_present
        * atom_j_present
        * not_padded
        * tf.expand_dims(template_valid, axis=1)
    )

    length_diff = tf.where(valid > 0.0, d_pred - d_true, tf.zeros_like(d_pred))
    sq_error = tf.square(length_diff)

    num = tf.reduce_sum(sq_error, axis=1)
    den = tf.reduce_sum(valid, axis=1) + eps
    per_sample = num / den
    return tf.reduce_mean(per_sample)


class CoordMainBondAuxModel(tf.keras.Model):
    """
    Custom training loop model with coordinate loss as main objective
    and bond-length loss as an auxiliary term.
    """

    def __init__(
        self,
        *args,
        bond_tables,
        bond_weight=0.25,
        residue_col,
        bead_col,
        residue_offset=0,
        num_res_types=19,
        max_beads=4,
        n_atoms=5,
        bond_scale=7.0,
        eps=1e-8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.bond_pairs_table, self.bond_valid_table, self.template_of_res_bead = bond_tables
        self.bond_weight = tf.Variable(
            float(bond_weight),
            trainable=False,
            dtype=tf.float32,
            name="bond_weight",
        )
        self.residue_col = int(residue_col)
        self.bead_col = int(bead_col)
        self.residue_offset = int(residue_offset)
        self.num_res_types = int(num_res_types)
        self.max_beads = int(max_beads)
        self.n_atoms = int(n_atoms)
        self.bond_scale = float(bond_scale)
        self.eps = float(eps)
        self.model_residue_scale = tf.constant(max(self.num_res_types - 1, 1), dtype=tf.float32)
        self.model_bead_scale = tf.constant(max(self.max_beads - 1, 1), dtype=tf.float32)

        self.coord_loss_fn = GeometryAwareMaskedMSE(n_atoms=self.n_atoms, eps=self.eps)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.bond_tracker = tf.keras.metrics.Mean(name="bond_mse")
        self.coord_tracker = GeometryAwareCoordMSE(n_atoms=self.n_atoms, eps=self.eps, name="coord_mse")
        self.rmse_tracker = GeometryAwareCoordRMSE(n_atoms=self.n_atoms, eps=self.eps, name="rmse")

    @property
    def metrics(self):
        return [self.loss_tracker, self.bond_tracker, self.coord_tracker, self.rmse_tracker]

    def _bond_loss(self, x, y_true, y_pred, mask):
        return bond_mse_from_tables(
            x=x,
            y_true=y_true,
            y_pred=y_pred,
            mask=mask,
            bond_pairs_table=self.bond_pairs_table,
            bond_valid_table=self.bond_valid_table,
            template_of_res_bead=self.template_of_res_bead,
            n_atoms=self.n_atoms,
            residue_col=self.residue_col,
            residue_offset=self.residue_offset,
            num_res_types=self.num_res_types,
            bead_col=self.bead_col,
            max_beads=self.max_beads,
            scale=self.bond_scale,
            eps=self.eps,
        )

    def _maybe_bond_loss(self, x, y_true, y_pred, mask):
        return tf.cond(
            tf.greater(self.bond_weight, 0.0),
            lambda: self._bond_loss(x, y_true, y_pred, mask),
            lambda: tf.constant(0.0, dtype=tf.float32),
        )

    def _model_input_from_raw(self, x):
        """
        Build model input from raw features:
        - keep all continuous features unchanged
        - scale residue/bead id columns into [0,1]-like ranges for the MLP
        Bond routing still uses the unscaled raw x.
        """
        x = tf.cast(x, tf.float32)

        res_col = self.residue_col
        bead_col = self.bead_col

        if res_col > bead_col:
            res_col, bead_col = bead_col, res_col
            swapped = True
        else:
            swapped = False

        left = x[:, :res_col]
        res_raw = x[:, res_col:res_col + 1]
        mid = x[:, res_col + 1:bead_col]
        bead_raw = x[:, bead_col:bead_col + 1]
        right = x[:, bead_col + 1:]

        res_scaled = tf.clip_by_value(
            (res_raw - float(self.residue_offset)) / self.model_residue_scale,
            0.0,
            1.0,
        )
        bead_scaled = tf.clip_by_value(bead_raw / self.model_bead_scale, 0.0, 1.0)

        if swapped:
            bead_scaled, res_scaled = res_scaled, bead_scaled

        return tf.concat([left, res_scaled, mid, bead_scaled, right], axis=1)

    def train_step(self, data):
        x, y_true, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        if sample_weight is None:
            sample_weight = tf.ones_like(y_true, dtype=tf.float32)

        with tf.GradientTape() as tape:
            x_model = self._model_input_from_raw(x)
            y_pred = self(x_model, training=True)
            coord_loss = self.coord_loss_fn(y_true, y_pred, sample_weight)
            bond_loss = self._maybe_bond_loss(x, y_true, y_pred, sample_weight)
            reg_loss = tf.add_n(self.losses) if self.losses else tf.constant(0.0, dtype=tf.float32)
            total_loss = coord_loss + (self.bond_weight * bond_loss) + reg_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(total_loss)
        self.bond_tracker.update_state(bond_loss)
        self.coord_tracker.update_state(y_true, y_pred, sample_weight=sample_weight)
        self.rmse_tracker.update_state(y_true, y_pred, sample_weight=sample_weight)

        return {
            "loss": self.loss_tracker.result(),
            "bond_mse": self.bond_tracker.result(),
            "coord_mse": self.coord_tracker.result(),
            "rmse": self.rmse_tracker.result(),
            "bond_weight": self.bond_weight,
        }

    def test_step(self, data):
        x, y_true, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        if sample_weight is None:
            sample_weight = tf.ones_like(y_true, dtype=tf.float32)

        x_model = self._model_input_from_raw(x)
        y_pred = self(x_model, training=False)
        coord_loss = self.coord_loss_fn(y_true, y_pred, sample_weight)
        bond_loss = self._maybe_bond_loss(x, y_true, y_pred, sample_weight)
        reg_loss = tf.add_n(self.losses) if self.losses else tf.constant(0.0, dtype=tf.float32)
        total_loss = coord_loss + (self.bond_weight * bond_loss) + reg_loss

        self.loss_tracker.update_state(total_loss)
        self.bond_tracker.update_state(bond_loss)
        self.coord_tracker.update_state(y_true, y_pred, sample_weight=sample_weight)
        self.rmse_tracker.update_state(y_true, y_pred, sample_weight=sample_weight)

        return {
            "loss": self.loss_tracker.result(),
            "bond_mse": self.bond_tracker.result(),
            "coord_mse": self.coord_tracker.result(),
            "rmse": self.rmse_tracker.result(),
            "bond_weight": self.bond_weight,
        }


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
    residue_col = int(os.getenv("RESIDUE_COL", str(RESIDUE_COL)))
    bead_col = int(os.getenv("BEAD_COL", str(BEAD_COL)))
    residue_offset = int(os.getenv("RESIDUE_OFFSET", "0"))
    bond_weight_target = float(os.getenv("BOND_WEIGHT", "0.25"))
    bond_weight_start = float(os.getenv("BOND_START_WEIGHT", "0.0"))
    coord_only_epochs = int(os.getenv("COORD_ONLY_EPOCHS", "20"))
    bond_ramp_epochs = int(os.getenv("BOND_RAMP_EPOCHS", "40"))
    bond_scale = float(os.getenv("BOND_SCALE", str(DEFAULT_BOND_SCALE)))
    bond_eps = float(os.getenv("BOND_EPS", str(DEFAULT_BOND_EPS)))
    cache_data_in_ram = _env_flag("CACHE_DATA_IN_RAM", default=False)
    batch_shuffle = _env_flag("BATCH_SHUFFLE", default=True)

    early_stop_patience = int(os.getenv("EARLY_STOP_PATIENCE", "25"))
    early_stop_min_delta = float(os.getenv("EARLY_STOP_MIN_DELTA", "1e-5"))
    reduce_lr_patience = int(os.getenv("REDUCE_LR_PATIENCE", "10"))
    min_lr = float(os.getenv("MIN_LR", "1e-6"))

    print("batch_size:", batch_size)
    print("epochs:", epochs)
    print("model_variant: single_output (coord main + weighted bond aux)")
    print("model_tag:", model_tag)
    print("model_dir:", model_dir)
    print("bond_lookup2 source:", BOND_LOOKUP2_PATH)
    print("residue_col/residue_offset:", residue_col, residue_offset)
    print("bead_col/max_beads:", bead_col, MAX_BEADS)
    print("bond_weight_start:", bond_weight_start)
    print("bond_weight_target:", bond_weight_target)
    print("coord_only_epochs:", coord_only_epochs)
    print("bond_ramp_epochs:", bond_ramp_epochs)
    print("model input scaling: residue_id/(num_res-1), bead_id/(max_beads-1); bond routing uses raw ids")
    print("bond_scale:", bond_scale)
    print("cache_data_in_ram:", cache_data_in_ram)
    print("batch_shuffle:", batch_shuffle)
    print("train rows (full):", x_train.shape[0])
    print("test rows  (full):", x_test.shape[0])
    print("the shape of X train is:", x_train.shape)
    print("the shape of Y train is as follows:", y_train.shape)
    print("the shape of Mask is as follows:", m_train.shape)

    if cache_data_in_ram:
        print("Caching train/test arrays to RAM as float32 for faster batch reads...")
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        m_train = np.asarray(m_train, dtype=np.float32)
        x_test = np.asarray(x_test, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)
        m_test = np.asarray(m_test, dtype=np.float32)
        print("RAM cache complete.")

    bond_pairs_table, bond_valid_table, template_of_res_bead, bond_stats = build_bond_tables(
        int_to_aa=BOND_INT_TO_AA,
        atom_order=BOND_ATOM_ORDER,
        atom_graph=BOND_ATOM_GRAPH,
        num_res=NUM_RES_TYPES,
        max_beads=MAX_BEADS,
    )
    print("Bond table stats:", bond_stats)
    bond_tables = (bond_pairs_table, bond_valid_table, template_of_res_bead)


    # Use streaming loaders so TensorFlow does not materialize full datasets as GPU constants.
    train_seq = MemmapBatchSequence(
        x=x_train,
        y=y_train,
        m=m_train,
        batch_size=batch_size,
        shuffle=batch_shuffle,
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

        base_model = build_single_output_model(
            trunk_width=512,
            trunk_blocks=trunk_blocks,
            latent_dim=512,
            dropout=dropout,
            l2_reg=l2_reg,
            cont_len=CONT_LEN,
            residue_col=residue_col,
            bead_col=bead_col,
            residue_vocab=NUM_RES_TYPES,
            residue_emb_dim=8,
            residue_offset=residue_offset,
            bead_vocab=MAX_BEADS,
            bead_emb_dim=4,
            bead_offset=0,
        )

        model = CoordMainBondAuxModel(
            inputs=base_model.inputs,
            outputs=base_model.outputs,
            name=base_model.name,
            bond_tables=bond_tables,
            bond_weight=bond_weight_start,
            residue_col=residue_col,
            bead_col=bead_col,
            residue_offset=residue_offset,
            num_res_types=NUM_RES_TYPES,
            max_beads=MAX_BEADS,
            n_atoms=N_ATOMS,
            bond_scale=bond_scale,
            eps=bond_eps,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr, clipnorm=1.0)
        model.compile(optimizer=optimizer)


    model.summary()

    ckpt_all = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, f"ckpt_epoch_{model_tag}_40_bond_{{epoch:02d}}.keras"),
        save_weights_only=False,
        save_freq="epoch",
        verbose=1,
    )

    ckpt_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, f"{model_tag}_40_bond_best.weights.h5"),
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
    bond_schedule = BondWeightScheduler(
        target_weight=bond_weight_target,
        coord_only_epochs=coord_only_epochs,
        ramp_epochs=bond_ramp_epochs,
        start_weight=bond_weight_start,
        verbose=1,
    )

    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[bond_schedule, ckpt_all, ckpt_best, early_stop, reduce_lr, nan_stop],
    )

    final_weights_path = os.path.join(model_dir, f"{model_tag}_bond_final.weights.h5")
    model.save_weights(final_weights_path)

    # Save a compile-free portable model for inference on other machines/clusters.
    portable_model = tf.keras.models.clone_model(base_model)
    portable_model.set_weights(model.get_weights())
    portable_model_path = os.path.join(model_dir, f"{model_tag}_bond_portable.keras")
    portable_model.save(portable_model_path)

    manifest_path = os.path.join(model_dir, f"{model_tag}_bond_manifest.json")
    manifest = {
        "model_tag": model_tag,
        "feature_dim": FEATURE_DIM,
        "target_dim": TARGET_DIM,
        "n_atoms": N_ATOMS,
        "loss_type": "coord_plus_weighted_bond",
        "bond_lookup2_path": BOND_LOOKUP2_PATH,
        "bond_table_stats": bond_stats,
        "model_dir": model_dir,
        "best_weights_path": os.path.join(model_dir, f"{model_tag}_bond_best.weights.h5"),
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
            "residue_col": residue_col,
            "bead_col": bead_col,
            "residue_offset": residue_offset,
            "num_res_types": NUM_RES_TYPES,
            "max_beads": MAX_BEADS,
            "bond_weight_start": bond_weight_start,
            "bond_weight_target": bond_weight_target,
            "coord_only_epochs": coord_only_epochs,
            "bond_ramp_epochs": bond_ramp_epochs,
            "bond_scale": bond_scale,
            "bond_eps": bond_eps,
            "cache_data_in_ram": cache_data_in_ram,
            "batch_shuffle": batch_shuffle,
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Saved model artifacts:", manifest_path)

    if "val_coord_mse" in history.history and history.history["val_coord_mse"]:
        best_index = int(np.argmin(history.history["val_coord_mse"]))
        train_coord = float(history.history["coord_mse"][best_index])
        val_coord = float(history.history["val_coord_mse"][best_index])
        train_bond = float(history.history["bond_mse"][best_index]) if "bond_mse" in history.history else float("nan")
        val_bond = float(history.history["val_bond_mse"][best_index]) if "val_bond_mse" in history.history else float("nan")
        train_rmse = float(history.history["rmse"][best_index]) if "rmse" in history.history else float("nan")
        val_rmse = float(history.history["val_rmse"][best_index]) if "val_rmse" in history.history else float("nan")
        print(f"Best epoch by val_coord_mse: {best_index + 1}")
        print(
            f"coord_mse train={train_coord:.6f}, val={val_coord:.6f} | "
            f"bond_mse train={train_bond:.6f}, val={val_bond:.6f} | "
            f"rmse train={train_rmse:.6f}, val={val_rmse:.6f}"
        )


if __name__ == "__main__":
    main()
