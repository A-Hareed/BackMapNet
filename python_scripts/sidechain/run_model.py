import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import argparse
import os
import itertools
from bond_lookup import INT_TO_AA, ATOM_ORDER, ATOM_GRAPH

MODEL_INPUT_DIM = 38
MODEL_CONT_LEN = 36
MODEL_NUM_EXPERTS = 18
MODEL_GATE_OFFSET = 1
MODEL_TRUNK_WIDTH = 512
MODEL_TRUNK_BLOCKS = 6
MODEL_FUSION_WIDTH = 512
MODEL_LATENT_DIM = 512
MODEL_EXPERT_HIDDEN = (256, 128)
MODEL_DROPOUT = 0.0
MODEL_BEAD_COL = 36
MODEL_BOND_WEIGHT = 2.0
MODEL_ANGLE_WEIGHT = 1.0
INT_TO_AA_18 = {i: INT_TO_AA[i + 1] for i in range(18)}



def atom_mse_norm_by_mask(n_atoms=5, eps=1e-8):
    # mask is per-coordinate, shape (B, 15)
    def loss(y_true, y_pred, mask):
        true = tf.reshape(y_true, (-1, n_atoms, 3))
        pred = tf.reshape(y_pred, (-1, n_atoms, 3))
        m    = tf.reshape(mask,   (-1, n_atoms, 3))

        se_atom = tf.reduce_sum(tf.square(true - pred) * m, axis=-1)  # (B, n_atoms)

        atom_valid = tf.reduce_any(m > 0.0, axis=-1)                  # (B, n_atoms)
        atom_valid_f = tf.cast(atom_valid, se_atom.dtype)

        num = tf.reduce_sum(se_atom * atom_valid_f, axis=-1)          # (B,)
        den = tf.reduce_sum(atom_valid_f, axis=-1) + eps              # (B,)
        return tf.reduce_mean(num / den)
    return loss

def atom_rmse_norm_by_mask(n_atoms=5, eps=1e-8):
    def rmse(y_true, y_pred, mask):
        true = tf.reshape(y_true, (-1, n_atoms, 3))
        pred = tf.reshape(y_pred, (-1, n_atoms, 3))
        m    = tf.reshape(mask,   (-1, n_atoms, 3))

        se_atom = tf.reduce_sum(tf.square(true - pred) * m, axis=-1)  # (B, n_atoms)

        atom_valid = tf.reduce_any(m > 0.0, axis=-1)                  # (B, n_atoms)
        atom_valid_f = tf.cast(atom_valid, se_atom.dtype)

        num = tf.reduce_sum(se_atom * atom_valid_f, axis=-1)          # (B,)
        den = tf.reduce_sum(atom_valid_f, axis=-1) + eps              # (B,)

        mse_per_sample  = num / den
        rmse_per_sample = tf.sqrt(mse_per_sample + eps)
        return tf.reduce_mean(rmse_per_sample)
    return rmse

def build_angle_tables(INT_TO_AA, ATOM_ORDER, ATOM_GRAPH, n_atoms=5):
    num_res = len(INT_TO_AA)
    max_beads = 0
    for aa in INT_TO_AA.values():
        if aa in ATOM_ORDER and len(ATOM_ORDER[aa]) > 0:
            max_beads = max(max_beads, max(ATOM_ORDER[aa].keys()) + 1)
    triplets = [[[] for _ in range(max_beads)] for _ in range(num_res)]
    for res_id, aa in INT_TO_AA.items():
        if aa not in ATOM_ORDER or aa not in ATOM_GRAPH:
            continue
        for bead_id, atom_list in ATOM_ORDER[aa].items():
            if bead_id not in ATOM_GRAPH[aa]:
                continue
            name_to_i = {name: i for i, name in enumerate(atom_list)}
            for center, neighs in ATOM_GRAPH[aa][bead_id]:
                if center not in name_to_i:
                    continue
                j = name_to_i[center]
                neigh_idx = [name_to_i[n] for n in neighs if n in name_to_i]
                for i, k in itertools.combinations(neigh_idx, 2):
                    if max(i, j, k) < n_atoms:
                        triplets[res_id][bead_id].append((i, j, k))
    max_angles = max((len(triplets[r][b]) for r in range(num_res) for b in range(max_beads)), default=0)
    angle_trip = -np.ones((num_res, max_beads, max_angles, 3), dtype=np.int32)
    angle_val = np.zeros((num_res, max_beads, max_angles), dtype=np.float32)
    for r in range(num_res):
        for b in range(max_beads):
            for a, t in enumerate(triplets[r][b]):
                angle_trip[r, b, a] = t
                angle_val[r, b, a] = 1.0
    return tf.constant(angle_trip, dtype=tf.int32), tf.constant(angle_val, dtype=tf.float32)

def build_bond_tables(INT_TO_AA, ATOM_ORDER, ATOM_GRAPH, num_res=18, max_beads=4):
    res_names = [INT_TO_AA[i] for i in range(num_res)]
    templates = []
    template_of_res_bead = -np.ones((num_res, max_beads), dtype=np.int32)
    for rid, res in enumerate(res_names):
        if res not in ATOM_ORDER:
            continue
        for bead in range(max_beads):
            if bead not in ATOM_ORDER[res] or res not in ATOM_GRAPH or bead not in ATOM_GRAPH[res]:
                continue
            atom_list = ATOM_ORDER[res][bead]
            atom_to_idx = {a: i for i, a in enumerate(atom_list)}
            bonds = set()
            for atom, neighs in ATOM_GRAPH[res][bead]:
                if atom not in atom_to_idx:
                    continue
                ia = atom_to_idx[atom]
                for nb in neighs:
                    if nb not in atom_to_idx:
                        continue
                    ib = atom_to_idx[nb]
                    if ia != ib:
                        a, b = (ia, ib) if ia < ib else (ib, ia)
                        bonds.add((a, b))
            template_id = len(templates)
            templates.append(sorted(list(bonds)))
            template_of_res_bead[rid, bead] = template_id
    max_bonds = max((len(b) for b in templates), default=1)
    T = len(templates)
    bond_pairs_table = -np.ones((T, max_bonds, 2), dtype=np.int32)
    bond_valid_table = np.zeros((T, max_bonds), dtype=np.float32)
    for t, bonds in enumerate(templates):
        for k, (i, j) in enumerate(bonds):
            bond_pairs_table[t, k, 0] = i
            bond_pairs_table[t, k, 1] = j
            bond_valid_table[t, k] = 1.0
    return (
        tf.constant(bond_pairs_table, dtype=tf.int32),
        tf.constant(bond_valid_table, dtype=tf.float32),
        tf.constant(template_of_res_bead, dtype=tf.int32),
    )

def bond_mse_from_tables(
    x, y_true, y_pred, mask,
    bond_pairs_table, bond_valid_table, template_of_res_bead,
    n_atoms=5, cont_len=36, gate_offset=1, num_experts=18, bead_col=36, max_beads=4, scale=7.0, eps=1e-8
):
    gate_raw = x[:, cont_len + 1]
    res_id = tf.cast(tf.round(tf.cast(gate_raw, tf.float32)), tf.int32) - gate_offset
    res_id = tf.clip_by_value(res_id, 0, num_experts - 1)
    bead_raw = x[:, bead_col]
    bead_id = tf.cast(tf.round(tf.cast(bead_raw, tf.float32)), tf.int32)
    bead_id = tf.clip_by_value(bead_id, 0, max_beads - 1)
    idx = tf.stack([res_id, bead_id], axis=1)
    tmpl = tf.gather_nd(template_of_res_bead, idx)
    tmpl_ok = tf.cast(tmpl >= 0, tf.float32)
    tmpl_safe = tf.maximum(tmpl, 0)
    pairs = tf.gather(bond_pairs_table, tmpl_safe)
    pair_valid = tf.gather(bond_valid_table, tmpl_safe)
    true = tf.reshape(y_true, (-1, n_atoms, 3))
    pred = tf.reshape(y_pred, (-1, n_atoms, 3))
    m = tf.reshape(mask, (-1, n_atoms, 3))
    atom_present = tf.cast(tf.reduce_any(m > 0.0, axis=-1), tf.float32)
    i, j = pairs[:, :, 0], pairs[:, :, 1]
    i_safe, j_safe = tf.maximum(i, 0), tf.maximum(j, 0)
    ti, tj = tf.gather(true, i_safe, batch_dims=1), tf.gather(true, j_safe, batch_dims=1)
    pi, pj = tf.gather(pred, i_safe, batch_dims=1), tf.gather(pred, j_safe, batch_dims=1)
    d_true = tf.sqrt(tf.reduce_sum(tf.square(ti - tj), axis=-1) + eps) * scale
    d_pred = tf.sqrt(tf.reduce_sum(tf.square(pi - pj), axis=-1) + eps) * scale
    ai, aj = tf.gather(atom_present, i_safe, batch_dims=1), tf.gather(atom_present, j_safe, batch_dims=1)
    not_padded = tf.cast((i >= 0) & (j >= 0), tf.float32)
    valid = pair_valid * ai * aj * not_padded * tf.expand_dims(tmpl_ok, axis=1)
    diff = tf.where(valid > 0.0, d_pred - d_true, tf.zeros_like(d_pred))
    num = tf.reduce_sum(tf.square(diff), axis=1)
    den = tf.reduce_sum(valid, axis=1) + eps
    return tf.reduce_mean(num / den)

def angle_mse_from_tables_stable(
    x, y_true, y_pred, mask,
    angle_triplets_table, angle_valid_table,
    n_atoms=5, cont_len=36, gate_offset=1, num_experts=18, bead_col=36, max_beads=4, eps=1e-8, min_len=2e-2
):
    gate_raw = x[:, cont_len + 1]
    gate_raw = tf.where(tf.math.is_finite(gate_raw), gate_raw, tf.zeros_like(gate_raw))
    res_id = tf.cast(tf.round(tf.cast(gate_raw, tf.float32)), tf.int32) - gate_offset
    res_id = tf.clip_by_value(res_id, 0, num_experts - 1)
    bead_raw = x[:, bead_col]
    bead_raw = tf.where(tf.math.is_finite(bead_raw), bead_raw, tf.zeros_like(bead_raw))
    bead_id = tf.cast(tf.round(tf.cast(bead_raw, tf.float32)), tf.int32)
    bead_id = tf.clip_by_value(bead_id, 0, max_beads - 1)
    true = tf.reshape(tf.cast(y_true, tf.float32), (-1, n_atoms, 3))
    pred = tf.reshape(tf.cast(y_pred, tf.float32), (-1, n_atoms, 3))
    m = tf.reshape(tf.cast(mask, tf.float32), (-1, n_atoms, 3))
    atom_valid = tf.reduce_any(m > 0.0, axis=-1)
    idx = tf.stack([res_id, bead_id], axis=1)
    trip = tf.gather_nd(angle_triplets_table, idx)
    base_valid = tf.gather_nd(angle_valid_table, idx)
    trip_ok = tf.reduce_all(trip >= 0, axis=-1)
    trip_safe = tf.maximum(trip, 0)
    i_idx, j_idx, k_idx = trip_safe[..., 0], trip_safe[..., 1], trip_safe[..., 2]
    vi = tf.gather(atom_valid, i_idx, batch_dims=1)
    vj = tf.gather(atom_valid, j_idx, batch_dims=1)
    vk = tf.gather(atom_valid, k_idx, batch_dims=1)
    valid = trip_ok & (base_valid > 0) & vi & vj & vk
    ti, tj, tk = tf.gather(true, i_idx, batch_dims=1), tf.gather(true, j_idx, batch_dims=1), tf.gather(true, k_idx, batch_dims=1)
    pi, pj, pk = tf.gather(pred, i_idx, batch_dims=1), tf.gather(pred, j_idx, batch_dims=1), tf.gather(pred, k_idx, batch_dims=1)
    v1t, v2t, v1p, v2p = ti - tj, tk - tj, pi - pj, pk - pj
    min_len2 = tf.cast(min_len * min_len, tf.float32)
    l1t2 = tf.reduce_sum(tf.square(v1t), axis=-1)
    l2t2 = tf.reduce_sum(tf.square(v2t), axis=-1)
    l1p2 = tf.reduce_sum(tf.square(v1p), axis=-1)
    l2p2 = tf.reduce_sum(tf.square(v2p), axis=-1)
    valid = valid & (l1t2 > min_len2) & (l2t2 > min_len2) & (l1p2 > min_len2) & (l2p2 > min_len2)
    inv1t = tf.math.rsqrt(tf.maximum(l1t2, min_len2))
    inv2t = tf.math.rsqrt(tf.maximum(l2t2, min_len2))
    inv1p = tf.math.rsqrt(tf.maximum(l1p2, min_len2))
    inv2p = tf.math.rsqrt(tf.maximum(l2p2, min_len2))
    u1t, u2t, u1p, u2p = v1t * inv1t[..., None], v2t * inv2t[..., None], v1p * inv1p[..., None], v2p * inv2p[..., None]
    cos_t = tf.clip_by_value(tf.reduce_sum(u1t * u2t, axis=-1), -1.0 + 1e-6, 1.0 - 1e-6)
    cos_p = tf.clip_by_value(tf.reduce_sum(u1p * u2p, axis=-1), -1.0 + 1e-6, 1.0 - 1e-6)
    sin_t = tf.norm(tf.linalg.cross(u1t, u2t), axis=-1)
    sin_p = tf.norm(tf.linalg.cross(u1p, u2p), axis=-1)
    per_ang = tf.square(cos_p - cos_t) + tf.square(sin_p - sin_t)
    per_ang = tf.where(valid, per_ang, tf.zeros_like(per_ang))
    num = tf.reduce_sum(per_ang, axis=-1)
    den = tf.reduce_sum(tf.cast(valid, tf.float32), axis=-1) + eps
    return tf.reduce_mean(num / den)



class MaskAbsorbingModel(tf.keras.Model):
    def __init__(
        self, *args, loss3=None, rmse3=None,
        bond_weight=0.0, bond_tables=None,
        angle_weight=0.0, angle_tables=None,
        bead_col=MODEL_BEAD_COL, n_atoms=5, cont_len=MODEL_CONT_LEN,
        gate_offset=MODEL_GATE_OFFSET, num_experts=MODEL_NUM_EXPERTS, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss3 = loss3
        self.rmse3 = rmse3
        self.bond_weight = tf.Variable(bond_weight, trainable=False, dtype=tf.float32, name="bond_w")
        self.angle_weight = tf.Variable(angle_weight, trainable=False, dtype=tf.float32, name="angle_w")
        self.bead_col = int(bead_col)
        self.n_atoms = int(n_atoms)
        self.cont_len = int(cont_len)
        self.gate_offset = int(gate_offset)
        self.num_experts = int(num_experts)
        self.bond_pairs_table, self.bond_valid_table, self.template_of_res_bead = bond_tables if bond_tables else (None, None, None)
        self.angle_triplets_table, self.angle_valid_table = angle_tables if angle_tables else (None, None)
        self.use_bond_loss = (self.bond_pairs_table is not None) and (float(bond_weight) > 0.0)
        self.use_angle_loss = (self.angle_triplets_table is not None) and (float(angle_weight) > 0.0)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.coord_tracker = tf.keras.metrics.Mean(name="coord_mse")
        self.rmse_tracker = tf.keras.metrics.Mean(name="rmse")
        self.bond_tracker = tf.keras.metrics.Mean(name="bond_mse")
        self.angle_tracker = tf.keras.metrics.Mean(name="angle_mse")

    @property
    def metrics(self):
        return [self.loss_tracker, self.coord_tracker, self.rmse_tracker, self.bond_tracker, self.angle_tracker]

    def test_step(self, data):
        x, y, sw = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_pred = self(x, training=False)
        coord_loss = self.loss3(y, y_pred, sw)
        bond_loss = tf.constant(0.0, dtype=tf.float32)
        if self.use_bond_loss:
            bond_loss = bond_mse_from_tables(
                x=x, y_true=y, y_pred=y_pred, mask=sw,
                bond_pairs_table=self.bond_pairs_table,
                bond_valid_table=self.bond_valid_table,
                template_of_res_bead=self.template_of_res_bead,
                n_atoms=self.n_atoms, cont_len=self.cont_len,
                gate_offset=self.gate_offset, num_experts=self.num_experts,
                bead_col=self.bead_col,
            )
        angle_loss = tf.constant(0.0, dtype=tf.float32)
        if self.use_angle_loss:
            angle_loss = angle_mse_from_tables_stable(
                x=x, y_true=y, y_pred=y_pred, mask=sw,
                angle_triplets_table=self.angle_triplets_table,
                angle_valid_table=self.angle_valid_table,
                n_atoms=self.n_atoms, cont_len=self.cont_len,
                gate_offset=self.gate_offset, num_experts=self.num_experts,
                bead_col=self.bead_col, max_beads=4, min_len=2e-2,
            )
        loss = coord_loss + self.bond_weight * bond_loss + self.angle_weight * angle_loss
        rmse = self.rmse3(y, y_pred, sw)
        self.loss_tracker.update_state(loss)
        self.coord_tracker.update_state(coord_loss)
        self.rmse_tracker.update_state(rmse)
        self.bond_tracker.update_state(bond_loss)
        self.angle_tracker.update_state(angle_loss)
        return {
            "loss": self.loss_tracker.result(),
            "coord_mse": self.coord_tracker.result(),
            "rmse": self.rmse_tracker.result(),
            "bond_mse": self.bond_tracker.result(),
            "angle_mse": self.angle_tracker.result(),
        }



def mlp_res_block(x, units=512, dropout=0.1, name="blk"):
    """Pre-LN residual MLP block for tabular features."""
    h = layers.LayerNormalization(name=f"{name}_ln")(x)
    h = layers.Dense(units, name=f"{name}_d0")(h)
    h = layers.LeakyReLU(0.2, name=f"{name}_a0")(h)
    if dropout and dropout > 0:
        h = layers.Dropout(dropout, name=f"{name}_drop0")(h)

    h = layers.Dense(units, name=f"{name}_d1")(h)

    # Residual projection if needed
    if x.shape[-1] != units:
        x = layers.Dense(units, name=f"{name}_proj")(x)

    out = layers.Add(name=f"{name}_add")([x, h])
    out = layers.LeakyReLU(0.2, name=f"{name}_a1")(out)
    return out




class HardGatedExperts(layers.Layer):
    def __init__(self, num_experts: int, y_dim: int, expert_hidden=(512, 256), alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = int(num_experts)
        self.y_dim = int(y_dim)
        self.alpha = float(alpha)

        self.experts = []
        for i in range(self.num_experts):
            seq = []
            for j, width in enumerate(expert_hidden):
                seq.append(layers.Dense(int(width), name=f"exp{i}_d{j}"))
                seq.append(layers.LeakyReLU(self.alpha, name=f"exp{i}_a{j}"))
            seq.append(layers.Dense(self.y_dim, name=f"exp{i}_out"))
            self.experts.append(tf.keras.Sequential(seq, name=f"expert_{i}"))

    def call(self, h, gate_id):
        # If gate_id came from float features, rounding avoids 17.999 -> 17 truncation surprises
        gate_id = tf.cast(tf.round(tf.cast(gate_id, tf.float32)), tf.int32)
        gate_id = tf.clip_by_value(gate_id, 0, self.num_experts - 1)

        B = tf.shape(h)[0]
        idx = tf.range(B)

        idx_parts = tf.dynamic_partition(idx, gate_id, self.num_experts)
        h_parts   = tf.dynamic_partition(h,   gate_id, self.num_experts)

        y_parts = []
        for i in range(self.num_experts):
            hi = h_parts[i]
            y_parts.append(
                tf.cond(
                    tf.shape(hi)[0] > 0,
                    lambda hi=hi, i=i: self.experts[i](hi),
                    lambda: tf.zeros((0, self.y_dim), dtype=h.dtype)
                )
            )

        return tf.dynamic_stitch(idx_parts, y_parts)








def build_model_tabular_sweetspot(
    input_dim=MODEL_INPUT_DIM,
    cont_len=MODEL_CONT_LEN,
    y_dim=15,
    num_experts=MODEL_NUM_EXPERTS,
    other_cat_vocab=4,
    other_cat_emb_dim=8,
    gate_offset=MODEL_GATE_OFFSET,
    gate_emb_dim=8,
    trunk_width=MODEL_TRUNK_WIDTH,
    trunk_blocks=MODEL_TRUNK_BLOCKS,
    fusion_width=MODEL_FUSION_WIDTH,
    latent_dim=MODEL_LATENT_DIM,
    expert_hidden=MODEL_EXPERT_HIDDEN,
    dropout=MODEL_DROPOUT,
    keep_mask_input=False,   # set True only if you still want mask as a model input
):
    # Inputs
    x_in = layers.Input(shape=(input_dim,), dtype=tf.float32, name="x")
    if keep_mask_input:
        mask_in = layers.Input(shape=(y_dim,), dtype=tf.float32, name="mask")

    # Slice continuous + categoricals
    cont = layers.Lambda(lambda t: t[:, :cont_len], name="cont_slice")(x_in)
    other_cat = layers.Lambda(lambda t: t[:, cont_len], name="cat25_slice")(x_in)
    gate_cat  = layers.Lambda(lambda t: t[:, cont_len + 1], name="gate26_slice")(x_in)

    # Cast categoricals to int
    other_cat_i = layers.Lambda(lambda t: tf.cast(t, tf.int32), name="cat25_int")(other_cat)
    gate_i = layers.Lambda(lambda t: tf.cast(t, tf.int32) - gate_offset, name="gate26_int")(gate_cat)
    gate_i = layers.Lambda(lambda t: tf.clip_by_value(t, 0, num_experts - 1), name="gate26_clip")(gate_i)

    # Embeddings
    other_e = layers.Embedding(other_cat_vocab, other_cat_emb_dim, name="emb_cat25")(other_cat_i)
    other_e = layers.Flatten(name="emb_cat25_flat")(other_e)

    gate_e = layers.Embedding(num_experts, gate_emb_dim, name="emb_gate26")(gate_i)
    gate_e = layers.Flatten(name="emb_gate26_flat")(gate_e)

    # ---- Tabular trunk (Dense-based) ----
    z = layers.LayerNormalization(name="cont_ln")(cont)
    z = layers.Dense(trunk_width, name="cont_d0")(z)
    z = layers.LeakyReLU(0.2, name="cont_a0")(z)
    if dropout and dropout > 0:
        z = layers.Dropout(dropout, name="cont_drop0")(z)

    for b in range(trunk_blocks):
        z = mlp_res_block(z, units=trunk_width, dropout=dropout, name=f"trunk_b{b}")

    # Fuse trunk + embeddings
    fused = layers.Concatenate(name="fuse")([z, other_e, gate_e])
    fused = layers.LayerNormalization(name="fuse_ln")(fused)
    fused = layers.Dense(fusion_width, name="fuse_d0")(fused)
    fused = layers.LeakyReLU(0.2, name="fuse_a0")(fused)
    if dropout and dropout > 0:
        fused = layers.Dropout(dropout, name="fuse_drop0")(fused)

    # Latent before experts
    h = layers.Dense(latent_dim, name="latent_d")(fused)
    h = layers.LeakyReLU(0.2, name="latent_a")(h)

    # Hard-gated experts -> (B, 15)
    yhat = HardGatedExperts(num_experts=num_experts, y_dim=y_dim, expert_hidden=expert_hidden, name="hard_route")(h, gate_i)

    # Optional: keep legacy mask input (NOT recommended if you mask via sample_weight in the loss)
    if keep_mask_input:
        yhat = layers.Multiply(name="yhat_masked")([yhat, mask_in])
        model = Model(inputs=[x_in, mask_in], outputs=yhat, name="HardGatedTabularMLP_SweetSpot")
    else:
        model = Model(inputs=x_in, outputs=yhat, name="HardGatedTabularMLP_SweetSpot")

    return model




# Bond/angle lookup tables (same residue-space convention as training.py)
bond_tables = build_bond_tables(
    INT_TO_AA=INT_TO_AA_18,
    ATOM_ORDER=ATOM_ORDER,
    ATOM_GRAPH=ATOM_GRAPH,
    num_res=18,
    max_beads=4,
)
angle_tables = build_angle_tables(INT_TO_AA_18, ATOM_ORDER, ATOM_GRAPH, n_atoms=5)

# 1) Build the base model EXACTLY as before
base = build_model_tabular_sweetspot(
        other_cat_vocab=4,
        num_experts=18,
        trunk_width=512,
        trunk_blocks=6,
        fusion_width=512,
        latent_dim=512,
        expert_hidden=(256, 128),
        dropout=0.0,           # for overfitting
        keep_mask_input=False  # if masking via sample_weight/loss
    )



model = MaskAbsorbingModel(
    inputs=base.inputs,
    outputs=base.outputs,
    loss3=atom_mse_norm_by_mask(n_atoms=5),
    rmse3=atom_rmse_norm_by_mask(n_atoms=5),
    bond_weight=MODEL_BOND_WEIGHT,
    bond_tables=bond_tables,
    angle_weight=MODEL_ANGLE_WEIGHT,
    angle_tables=angle_tables,
    bead_col=MODEL_BEAD_COL,
    n_atoms=5,
    cont_len=MODEL_CONT_LEN,
    gate_offset=MODEL_GATE_OFFSET,
    num_experts=MODEL_NUM_EXPERTS,
)
print(model.weights)

# Important: build variables by calling once (or model.build)
_ = model(tf.zeros((1, 38), dtype=tf.float32), training=False)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run side-chain model with dynamic feature/target/mask inputs."
    )
    parser.add_argument("feature_file", nargs="?", help="Feature file (.npy or .npz)")
    parser.add_argument("target_file", nargs="?", help="Target file (.npy or .npz)")
    parser.add_argument("mask_file", nargs="?", help="Mask file (.npy or .npz)")
    parser.add_argument("cluster", nargs="?", help="Cluster id (optional; used for output naming)")
    parser.add_argument("--pdb", default=None, help="PDB id for default file naming")
    parser.add_argument(
        "--weights",
        default="EXPERT_M24_best.weights.h5",
        help="Weights file (.h5/.weights.h5) or full Keras model file (.keras).",
    )
    parser.add_argument("--out", default=None, help="Output prediction .npy filename")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation step")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for evaluate/predict.")
    parser.add_argument(
        "--filter-residue-id",
        type=int,
        default=None,
        help="If set, drop rows where rounded X[:, gate_col] == this residue id.",
    )
    parser.add_argument(
        "--gate-col",
        type=int,
        default=37,
        help="Gate/residue-id column in X used for filtering (default: 37).",
    )
    return parser.parse_args()


def load_arr(path, mmap_npy=True):
    if path.endswith(".npz"):
        arr = np.load(path)
        if "arr" not in arr:
            raise ValueError(f"{path} is .npz but missing key 'arr'")
        return arr["arr"].astype(np.float32)
    if mmap_npy:
        arr = np.load(path, mmap_mode="r")
        if arr.dtype != np.float32:
            return np.asarray(arr, dtype=np.float32)
        return arr
    return np.load(path).astype(np.float32)


def load_runtime_model_weights(runtime_model, source_path):
    """
    Load model parameters into runtime_model from either:
    1) a weights file (default path), or
    2) a full .keras model file (weights are transferred via set_weights).
    """
    try:
        runtime_model.load_weights(source_path)
        print(f"Loaded side-chain weights: {source_path}")
        return
    except Exception as weight_error:
        if not source_path.endswith(".keras"):
            raise ValueError(
                f"Failed to load side-chain weights from: {source_path}\n"
                f"Original error: {weight_error}"
            ) from weight_error

        custom_objects = {
            "HardGatedExperts": HardGatedExperts,
            "MaskAbsorbingModel": MaskAbsorbingModel,
            "atom_mse_norm_by_mask": atom_mse_norm_by_mask,
            "atom_rmse_norm_by_mask": atom_rmse_norm_by_mask,
        }
        try:
            loaded_model = tf.keras.models.load_model(
                source_path,
                custom_objects=custom_objects,
                compile=False,
            )
            runtime_model.set_weights(loaded_model.get_weights())
            print(f"Loaded side-chain full model and transferred weights: {source_path}")
        except Exception as model_error:
            raise ValueError(
                f"Failed to load side-chain model from: {source_path}\n"
                f"Weight-load error: {weight_error}\n"
                f"Full-model transfer error: {model_error}"
            ) from model_error


def get_first_dim(path):
    if path.endswith(".npz"):
        arr = np.load(path)["arr"]
    else:
        arr = np.load(path, mmap_mode="r")
    return int(arr.shape[0])


def summarize_filter(X, residue_id, gate_col=26, name="data"):
    if X.ndim != 2:
        raise ValueError(f"Expected 2D X for filtering, got shape {X.shape}")
    if gate_col < 0 or gate_col >= X.shape[1]:
        raise ValueError(f"gate_col={gate_col} out of range for X shape {X.shape}")

    gate = np.rint(X[:, gate_col]).astype(np.int32)
    keep = gate != int(residue_id)

    n = gate.shape[0]
    kept = int(keep.sum())
    removed = n - kept

    print(f"[{name}] total: {n:,}")
    print(f"[{name}] keep : {kept:,} ({kept/n:.2%})")
    print(f"[{name}] drop : {removed:,} ({removed/n:.2%})")

    if kept == 0:
        raise RuntimeError(
            "Filter kept 0 rows. residue_id may be wrong or gate column may be incorrect."
        )
    if removed == 0:
        print("Warning: removed 0 rows. No matches found for filter-residue-id.")

    if removed > 0:
        removed_ids = np.unique(gate[~keep])
        suffix = "..." if removed_ids.size > 20 else ""
        print(f"[{name}] unique gate ids removed: {removed_ids[:20]}{suffix}")

    vals, cnts = np.unique(gate, return_counts=True)
    top = np.argsort(-cnts)[:10]
    print(f"[{name}] top gate ids: {list(zip(vals[top], cnts[top]))}")
    return keep


args = parse_args()

if args.feature_file and args.target_file and args.mask_file:
    feature_path = args.feature_file
    target_path = args.target_file
    mask_path = args.mask_file
else:
    if args.pdb is None:
        raise ValueError(
            "Provide either <feature_file> <target_file> <mask_file> or --pdb for default names."
        )
    feature_path = f"feature_SideChain_testing_{args.pdb}_subset.npz"
    target_path = f"target_SideChain_testing_{args.pdb}_subset.npz"
    mask_path = f"input_Masking_testing_{args.pdb}_localFrame.npz"

for p in (feature_path, target_path, mask_path, args.weights):
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing required file: {p}")

# 3) Load weights and compile
model.summary()
load_runtime_model_weights(model, args.weights)
print(model.weights)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4, clipnorm=1.0))

print("Using inputs:")
print("  feature:", feature_path)
print("  target :", target_path)
print("  mask   :", mask_path)
print("  batch_size:", args.batch_size)

X = load_arr(feature_path, mmap_npy=True)
Y = None
M = None
Y_orig_shape = None
M_orig_shape = None
group_keep_idx = None

if X.ndim != 2 or X.shape[1] != 38:
    raise ValueError(f"Expected feature array shape (*, {MODEL_INPUT_DIM}), got {X.shape}")

# Only load Y/M when needed for eval.
if not args.no_eval:
    Y = load_arr(target_path, mmap_npy=True)
    M = load_arr(mask_path, mmap_npy=True)
    Y_orig_shape = Y.shape
    M_orig_shape = M.shape

    # Align target/mask shape to flattened feature rows when needed.
    if X.ndim == 2 and Y.ndim == 2 and M.ndim == 2:
        if X.shape[0] != Y.shape[0] and Y.shape[1] % 15 == 0:
            Y = Y.reshape(-1, 15).astype(np.float32)
        if X.shape[0] != M.shape[0] and M.shape[1] % 15 == 0:
            M = M.reshape(-1, 15).astype(np.float32)

    if M.ndim == 2 and np.any(M < 0):
        # Convert legacy -2 padded masks into binary sample weights.
        M = (M > 0).astype(np.float32)

    if not (X.shape[0] == Y.shape[0] == M.shape[0]):
        raise ValueError(
            f"Sample mismatch after reshape: X={X.shape}, Y={Y.shape}, M={M.shape}"
        )
else:
    # Keep original sample count for reshape output without loading full Y/M.
    Y_orig_shape = (get_first_dim(target_path),)
    M_orig_shape = (get_first_dim(mask_path),)

if args.filter_residue_id is not None:
    keep = summarize_filter(
        X, residue_id=args.filter_residue_id, gate_col=args.gate_col, name="inference"
    )
    old_sample_num = int(M_orig_shape[0]) if M_orig_shape is not None and len(M_orig_shape) >= 1 else 0
    if old_sample_num > 0 and keep.size % old_sample_num == 0:
        groups_per_sample = keep.size // old_sample_num
        keep_2d = keep.reshape(old_sample_num, groups_per_sample)
        if np.all(keep_2d == keep_2d[0]):
            group_keep = keep_2d[0]
        else:
            # Conservative: keep only groups retained in all samples.
            group_keep = np.all(keep_2d, axis=0)
            print("Warning: filter mask varies by sample; using intersection across samples.")
        group_keep_idx = np.where(group_keep)[0].astype(np.int32)
        np.save("expert_filter_keep_group_idx.npy", group_keep_idx)
        print("Saved: expert_filter_keep_group_idx.npy", group_keep_idx.shape)
    else:
        print(
            "Warning: could not derive group keep index from filter mask.",
            f"keep_size={keep.size}, old_sample_num={old_sample_num}",
        )

    X = np.asarray(X[keep], dtype=np.float32)
    if not args.no_eval:
        Y = np.asarray(Y[keep], dtype=np.float32)
        M = np.asarray(M[keep], dtype=np.float32)
        print("Shapes after filter:", X.shape, Y.shape, M.shape)
    else:
        print("Shapes after filter:", X.shape)

if not args.no_eval:
    print(X.shape, Y.shape, M.shape)
else:
    print(X.shape)

if not args.no_eval:
    model.evaluate(X, Y, sample_weight=M, batch_size=args.batch_size, verbose=1)

yhat = model.predict(X, batch_size=args.batch_size, verbose=1)
print("shape of the prediction", yhat.shape)

if args.out:
    out_file = args.out
elif args.cluster is not None:
    out_file = f"expert_{args.cluster}_Yhat.npy"
elif args.pdb is not None:
    out_file = f"expert_{args.pdb}_Yhat.npy"
else:
    out_file = "expert_Yhat.npy"

np.save(out_file, yhat)
print("Saved:", out_file)

# Save a frame-wise reshaped prediction for downstream denorm scripts.
# Target shape is (old_sample_num, -1) where old_sample_num comes from original Y/M.
if yhat.ndim == 2:
    old_sample_num = None
    if Y_orig_shape is not None and len(Y_orig_shape) >= 1:
        old_sample_num = int(Y_orig_shape[0])
    if M_orig_shape is not None and len(M_orig_shape) >= 1 and int(M_orig_shape[0]) > 0:
        if old_sample_num is None or old_sample_num <= 0:
            old_sample_num = int(M_orig_shape[0])

    total_vals = yhat.size
    if old_sample_num and total_vals % old_sample_num == 0:
        yhat_reshaped = yhat.reshape(old_sample_num, -1).astype(np.float32)
        np.save("expert_Yhat_reshaped.npy", yhat_reshaped)
        print("Saved: expert_Yhat_reshaped.npy", yhat_reshaped.shape)
    else:
        print(
            "Warning: could not reshape prediction to (old_sample_num, -1).",
            f"old_sample_num={old_sample_num}, pred_size={total_vals}",
        )
