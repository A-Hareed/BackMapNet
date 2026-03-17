import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import glob
import os
from numpy.lib.format import open_memmap

import itertools
import numpy as np
from bond_lookup import INT_TO_AA, ATOM_ORDER, ATOM_GRAPH
import re 

# Avoid eager full GPU pre-allocation.
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except Exception:
        pass

INT_TO_AA_18 = {i: INT_TO_AA[i+1] for i in range(18)}

def build_angle_tables(INT_TO_AA, ATOM_ORDER, ATOM_GRAPH, n_atoms=5):
    num_res = len(INT_TO_AA)

    # infer max beads from ATOM_ORDER
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

            # map atom name -> local index
            name_to_i = {name: i for i, name in enumerate(atom_list)}
            # build angles from (center, neighbors) definitions
            for center, neighs in ATOM_GRAPH[aa][bead_id]:
                if center not in name_to_i:
                    continue
                j = name_to_i[center]
                neigh_idx = [name_to_i[n] for n in neighs if n in name_to_i]

                # need pairs of neighbors to define an angle
                for i, k in itertools.combinations(neigh_idx, 2):
                    if max(i, j, k) < n_atoms:
                        triplets[res_id][bead_id].append((i, j, k))

    max_angles = max((len(triplets[r][b])
                      for r in range(num_res)
                      for b in range(max_beads)), default=0)

    angle_trip = -np.ones((num_res, max_beads, max_angles, 3), dtype=np.int32)
    angle_val  = np.zeros((num_res, max_beads, max_angles), dtype=np.float32)

    for r in range(num_res):
        for b in range(max_beads):
            for a, t in enumerate(triplets[r][b]):
                angle_trip[r, b, a] = t
                angle_val[r, b, a] = 1.0

    return angle_trip, angle_val



def build_bond_tables(INT_TO_AA, ATOM_ORDER, ATOM_GRAPH, num_res=19, max_beads=4):
    """
    Returns:
      bond_pairs_table: (T, max_bonds, 2) int32, padded with -1
      bond_valid_table: (T, max_bonds) float32
      template_of_res_bead: (num_res, max_beads) int32, -1 if invalid
    """
    # Map residue_id -> residue_name for the experts you actually train (0..18)
    res_names = [INT_TO_AA[i] for i in range(num_res)]

    templates = []
    template_of_res_bead = -np.ones((num_res, max_beads), dtype=np.int32)

    # Collect bond pairs per (res, bead)
    for rid, res in enumerate(res_names):
        if res not in ATOM_ORDER:
            continue

        for bead in range(max_beads):
            if bead not in ATOM_ORDER[res]:
                continue
            if res not in ATOM_GRAPH or bead not in ATOM_GRAPH[res]:
                continue

            atom_list = ATOM_ORDER[res][bead]              # e.g. ["CB","CG","CD1"...]
            atom_to_idx = {a:i for i,a in enumerate(atom_list)}

            # Build unique undirected bonds inside this bead
            bonds = set()
            for atom, neighs in ATOM_GRAPH[res][bead]:
                if atom not in atom_to_idx:
                    continue
                ia = atom_to_idx[atom]
                for nb in neighs:
                    if nb not in atom_to_idx:
                        continue
                    ib = atom_to_idx[nb]
                    if ia == ib:
                        continue
                    a, b = (ia, ib) if ia < ib else (ib, ia)
                    bonds.add((a, b))

            bonds = sorted(list(bonds))
            template_id = len(templates)
            templates.append(bonds)
            template_of_res_bead[rid, bead] = template_id

    # Pad to rectangular tensors
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


#-------------define bond dictionary-------------------------------------
bond_pairs_table, bond_valid_table, template_of_res_bead = build_bond_tables(
    INT_TO_AA=INT_TO_AA_18,
    ATOM_ORDER=ATOM_ORDER,
    ATOM_GRAPH=ATOM_GRAPH,
    num_res=18,      # 0..18 (no GLY)
    max_beads=4
)
# angle
angle_trip_np, angle_val_np = build_angle_tables(INT_TO_AA_18, ATOM_ORDER, ATOM_GRAPH, n_atoms=5)
angle_tables = (
    tf.constant(angle_trip_np, dtype=tf.int32),
    tf.constant(angle_val_np,  dtype=tf.float32),
)
#----------------------------------------------------------



def angle_mse_from_tables_stable(
    x, y_true, y_pred, mask,
    angle_triplets_table, angle_valid_table,
    n_atoms=5,
    cont_len=36,
    gate_offset=1,
    num_experts=18,
    bead_col=36,
    max_beads=4,
    eps=1e-8,
    min_len=2e-2,   # normalized coords: start 0.02; try 0.03–0.05 if needed
):
    """
    Robust angle loss using (cos, sin) representation:
      L = (cos_p - cos_t)^2 + (sin_p - sin_t)^2  in [0,4]
    No atan2 -> avoids NaN gradients from atan2(0,0).
    """
    # Extract x tensor
    if isinstance(x, dict):
        x_t = x.get("x", next(iter(x.values())))
    elif isinstance(x, (list, tuple)):
        x_t = x[0]
    else:
        x_t = x

    # residue id
    gate_raw = x_t[:, cont_len + 1]
    gate_raw = tf.where(tf.math.is_finite(gate_raw), gate_raw, tf.zeros_like(gate_raw))
    res_id = tf.cast(tf.round(tf.cast(gate_raw, tf.float32)), tf.int32) - gate_offset
    res_id = tf.clip_by_value(res_id, 0, num_experts - 1)

    # bead id
    bead_raw = x_t[:, bead_col]
    bead_raw = tf.where(tf.math.is_finite(bead_raw), bead_raw, tf.zeros_like(bead_raw))
    bead_id = tf.cast(tf.round(tf.cast(bead_raw, tf.float32)), tf.int32)
    bead_id = tf.clip_by_value(bead_id, 0, max_beads - 1)

    # Reshape coords
    true = tf.reshape(tf.cast(y_true, tf.float32), (-1, n_atoms, 3))
    pred = tf.reshape(tf.cast(y_pred, tf.float32), (-1, n_atoms, 3))
    m    = tf.reshape(tf.cast(mask,   tf.float32), (-1, n_atoms, 3))

    atom_valid = tf.reduce_any(m > 0.0, axis=-1)  # (B, n_atoms) bool

    # Lookup triplets
    idx = tf.stack([res_id, bead_id], axis=1)  # (B,2)
    trip = tf.gather_nd(angle_triplets_table, idx)     # (B, A, 3) padded -1
    base_valid = tf.gather_nd(angle_valid_table, idx)  # (B, A)

    trip_ok = tf.reduce_all(trip >= 0, axis=-1)        # (B, A) bool
    trip_safe = tf.maximum(trip, 0)

    i_idx = trip_safe[..., 0]
    j_idx = trip_safe[..., 1]
    k_idx = trip_safe[..., 2]

    # atom existence
    vi = tf.gather(atom_valid, i_idx, batch_dims=1)
    vj = tf.gather(atom_valid, j_idx, batch_dims=1)
    vk = tf.gather(atom_valid, k_idx, batch_dims=1)

    valid = trip_ok & (base_valid > 0) & vi & vj & vk  # (B, A) bool

    # gather coords
    ti = tf.gather(true, i_idx, batch_dims=1)
    tj = tf.gather(true, j_idx, batch_dims=1)
    tk = tf.gather(true, k_idx, batch_dims=1)

    pi = tf.gather(pred, i_idx, batch_dims=1)
    pj = tf.gather(pred, j_idx, batch_dims=1)
    pk = tf.gather(pred, k_idx, batch_dims=1)

    # vectors
    v1t = ti - tj
    v2t = tk - tj
    v1p = pi - pj
    v2p = pk - pj

    # squared lengths
    min_len2 = tf.cast(min_len * min_len, tf.float32)
    l1t2 = tf.reduce_sum(tf.square(v1t), axis=-1)
    l2t2 = tf.reduce_sum(tf.square(v2t), axis=-1)
    l1p2 = tf.reduce_sum(tf.square(v1p), axis=-1)
    l2p2 = tf.reduce_sum(tf.square(v2p), axis=-1)

    # validity: non-degenerate in BOTH true and pred
    valid = valid & (l1t2 > min_len2) & (l2t2 > min_len2) & (l1p2 > min_len2) & (l2p2 > min_len2)

    # normalize with rsqrt(max(len2, min_len2))
    inv1t = tf.math.rsqrt(tf.maximum(l1t2, min_len2))
    inv2t = tf.math.rsqrt(tf.maximum(l2t2, min_len2))
    inv1p = tf.math.rsqrt(tf.maximum(l1p2, min_len2))
    inv2p = tf.math.rsqrt(tf.maximum(l2p2, min_len2))

    u1t = v1t * inv1t[..., None]
    u2t = v2t * inv2t[..., None]
    u1p = v1p * inv1p[..., None]
    u2p = v2p * inv2p[..., None]

    # cos and sin of angles (bounded, smooth)
    cos_t = tf.reduce_sum(u1t * u2t, axis=-1)
    cos_p = tf.reduce_sum(u1p * u2p, axis=-1)
    cos_t = tf.clip_by_value(cos_t, -1.0 + 1e-6, 1.0 - 1e-6)
    cos_p = tf.clip_by_value(cos_p, -1.0 + 1e-6, 1.0 - 1e-6)

    sin_t = tf.norm(tf.linalg.cross(u1t, u2t), axis=-1)
    sin_p = tf.norm(tf.linalg.cross(u1p, u2p), axis=-1)

    # circular angle distance in (cos,sin) space: bounded [0,4]
    per_ang = tf.square(cos_p - cos_t) + tf.square(sin_p - sin_t)

    # kill invalid entries BEFORE reduction (prevents NaN leakage)
    per_ang = tf.where(valid, per_ang, tf.zeros_like(per_ang))

    num = tf.reduce_sum(per_ang, axis=-1)  # (B,)
    den = tf.reduce_sum(tf.cast(valid, tf.float32), axis=-1) + eps
    return tf.reduce_mean(num / den)



def bond_mse_from_tables(
    x, y_true, y_pred, mask,
    bond_pairs_table, bond_valid_table, template_of_res_bead,
    n_atoms=5, cont_len=36, gate_offset=1, num_experts=18,
    bead_col=None,   # <-- set this!
    max_beads=4,scale=7.0,
    eps=1e-8
):
    """
    x: (B, input_dim)
    y_true/y_pred/mask: (B, 15) where 15 = 5 atoms * 3 coords
    bead_col: column index in x for bead_id (0..max_beads-1)
    """
    if bead_col is None:
        raise ValueError("You must set bead_col to the column index in x that stores bead_id (0..3).")

    # Extract residue_id (gate) and bead_id
    gate_raw = x[:, cont_len + 1]
    res_id = tf.cast(tf.round(tf.cast(gate_raw, tf.float32)), tf.int32) - gate_offset
    res_id = tf.clip_by_value(res_id, 0, num_experts - 1)

    bead_raw = x[:, bead_col]
    bead_id = tf.cast(tf.round(tf.cast(bead_raw, tf.float32)), tf.int32)
    bead_id = tf.clip_by_value(bead_id, 0, max_beads - 1)

    # Map (res_id, bead_id) -> template_id
    idx = tf.stack([res_id, bead_id], axis=1)  # (B, 2)
    tmpl = tf.gather_nd(template_of_res_bead, idx)  # (B,)
    tmpl_ok = tf.cast(tmpl >= 0, tf.float32)

    # Get bond pairs for each sample: (B, max_bonds, 2)
    # For invalid tmpl (-1), replace with 0 safely; we'll mask it out with tmpl_ok
    tmpl_safe = tf.maximum(tmpl, 0)
    pairs = tf.gather(bond_pairs_table, tmpl_safe)      # (B, K, 2)
    pair_valid = tf.gather(bond_valid_table, tmpl_safe) # (B, K)

    # Reshape coords
    true = tf.reshape(y_true, (-1, n_atoms, 3))
    pred = tf.reshape(y_pred, (-1, n_atoms, 3))
    m    = tf.reshape(mask,   (-1, n_atoms, 3))

    # Atom present? (B, n_atoms)
    atom_present = tf.cast(tf.reduce_any(m > 0.0, axis=-1), tf.float32)

    # Bond indices
    i = pairs[:, :, 0]  # (B, K)
    j = pairs[:, :, 1]  # (B, K)

    # Safe indices for gather; padded pairs are -1
    i_safe = tf.maximum(i, 0)
    j_safe = tf.maximum(j, 0)

    # Gather coords: (B, K, 3)
    ti = tf.gather(true, i_safe, batch_dims=1)
    tj = tf.gather(true, j_safe, batch_dims=1)
    pi = tf.gather(pred, i_safe, batch_dims=1)
    pj = tf.gather(pred, j_safe, batch_dims=1)

    # Distances: (B, K)
    d_true = tf.sqrt(tf.reduce_sum(tf.square(ti - tj), axis=-1) + eps)
    d_pred = tf.sqrt(tf.reduce_sum(tf.square(pi - pj), axis=-1) + eps)
    # Convert to Å
    d_true = d_true * scale
    d_pred = d_pred * scale
    # Bond exists only if:
    # - bond is valid in template
    # - both atoms are present in this sample
    # - template is valid for this (res, bead)
    ai = tf.gather(atom_present, i_safe, batch_dims=1)
    aj = tf.gather(atom_present, j_safe, batch_dims=1)

    # kill padded bonds where i/j was -1
    not_padded = tf.cast((i >= 0) & (j >= 0), tf.float32)

    valid = pair_valid * ai * aj * not_padded * tf.expand_dims(tmpl_ok, axis=1)


    diff = d_pred - d_true
    diff = tf.where(valid > 0.0, diff, tf.zeros_like(diff))  # prevents NaN*0
    se = tf.square(diff)

    #se = tf.square(d_pred - d_true) * valid  # (B, K)



    num = tf.reduce_sum(se, axis=1)
    den = tf.reduce_sum(valid, axis=1) + eps
    per_sample = num / den
    return tf.reduce_mean(per_sample)

def angle_mse_norm_by_mask(angle_idx_table, n_atoms=5, eps=1e-8):
    """
    angle_idx_table: tf.int32 tensor (num_res, max_beads, max_angles, 3) with -1 padding
    res_id: (B,) int32
    bead_id: (B,) int32
    """
    def angle_from_points(p_i, p_j, p_k):
        # angle at j between vectors (i-j) and (k-j)
        v1 = p_i - p_j
        v2 = p_k - p_j
        # atan2(||cross||, dot) is stable
        cross = tf.linalg.cross(v1, v2)
        num = tf.norm(cross, axis=-1)
        den = tf.reduce_sum(v1 * v2, axis=-1)
        return tf.atan2(num, den)  # (B, max_angles)

    def loss(y_true, y_pred, mask, res_id, bead_id):
        true = tf.reshape(y_true, (-1, n_atoms, 3))
        pred = tf.reshape(y_pred, (-1, n_atoms, 3))
        m    = tf.reshape(mask,   (-1, n_atoms, 3))

        atom_valid = tf.reduce_any(m > 0.0, axis=-1)  # (B, n_atoms) bool

        # lookup triplets for each sample
        idx = tf.stack([res_id, bead_id], axis=1)                      # (B,2)
        trip = tf.gather_nd(angle_idx_table, idx)                      # (B, max_angles, 3)
        trip_ok = tf.reduce_all(trip >= 0, axis=-1)                    # (B, max_angles) bool
        trip_safe = tf.maximum(trip, 0)                                # avoid gather(-1)

        i_idx = trip_safe[..., 0]
        j_idx = trip_safe[..., 1]
        k_idx = trip_safe[..., 2]

        # gather coords: (B, max_angles, 3)
        pi_t = tf.gather(true, i_idx, batch_dims=1)
        pj_t = tf.gather(true, j_idx, batch_dims=1)
        pk_t = tf.gather(true, k_idx, batch_dims=1)

        pi_p = tf.gather(pred, i_idx, batch_dims=1)
        pj_p = tf.gather(pred, j_idx, batch_dims=1)
        pk_p = tf.gather(pred, k_idx, batch_dims=1)

        # validity: triplet exists AND all 3 atoms exist
        vi = tf.gather(atom_valid, i_idx, batch_dims=1)
        vj = tf.gather(atom_valid, j_idx, batch_dims=1)
        vk = tf.gather(atom_valid, k_idx, batch_dims=1)
        ang_valid = trip_ok & vi & vj & vk
        ang_valid_f = tf.cast(ang_valid, tf.float32)

        # compute angles
        ang_t = angle_from_points(pi_t, pj_t, pk_t)
        ang_p = angle_from_points(pi_p, pj_p, pk_p)

        # periodic-safe difference (wrap to [-pi, pi])
        d = tf.atan2(tf.sin(ang_p - ang_t), tf.cos(ang_p - ang_t))
        se = tf.square(d) * ang_valid_f

        num = tf.reduce_sum(se, axis=-1)                # (B,)
        den = tf.reduce_sum(ang_valid_f, axis=-1) + eps # (B,)
        return tf.reduce_mean(num / den)

    return loss


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



class MaskAbsorbingModel(tf.keras.Model):
    def __init__(self, *args,
                 loss3=None, rmse3=None,
                 bond_weight=0.0,
                 bond_tables=None,   # (bond_pairs_table, bond_valid_table, template_of_res_bead)
                 angle_weight=0.0,
                 angle_tables=None,  # (angle_triplets_table, angle_valid_table)
                 bead_col=None,
                 n_atoms=5, cont_len=36, gate_offset=1, num_experts=18,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.loss3 = loss3
        #self.coord_tracker = tf.keras.metrics.Mean(name="coord_mse")
        self.rmse3 = rmse3

#        self.bond_weight = float(bond_weight)
#        self.angle_weight = float(angle_weight)


        self.bond_weight  = tf.Variable(bond_weight, trainable=False, dtype=tf.float32, name="bond_w")
        self.angle_weight = tf.Variable(angle_weight, trainable=False, dtype=tf.float32, name="angle_w")




        self.bead_col = bead_col
        self.n_atoms = int(n_atoms)
        self.cont_len = int(cont_len)
        self.gate_offset = int(gate_offset)
        self.num_experts = int(num_experts)

        self.bond_pairs_table, self.bond_valid_table, self.template_of_res_bead = bond_tables if bond_tables else (None, None, None)
        self.angle_triplets_table, self.angle_valid_table = angle_tables if angle_tables else (None, None)
        self.use_bond_loss = (self.bond_pairs_table is not None) and (float(bond_weight) > 0.0)
        self.use_angle_loss = (self.angle_triplets_table is not None) and (float(angle_weight) > 0.0)

        self.loss_tracker  = tf.keras.metrics.Mean(name="loss")
        self.coord_tracker = tf.keras.metrics.Mean(name="coord_mse")
        self.rmse_tracker  = tf.keras.metrics.Mean(name="rmse")
        self.bond_tracker  = tf.keras.metrics.Mean(name="bond_mse")
        self.angle_tracker = tf.keras.metrics.Mean(name="angle_mse")

    @property
    def metrics(self):
        return [self.loss_tracker, self.coord_tracker, self.rmse_tracker, self.bond_tracker, self.angle_tracker]

    def _routing_assertions(self, x):
        x = tf.cast(x, tf.float32)
        tf.debugging.assert_greater_equal(
            tf.shape(x)[1], self.cont_len + 2,
            message="Input feature width is too small for cont_len/gate columns."
        )
        gate_raw = tf.cast(tf.round(x[:, self.cont_len + 1]), tf.int32)
        bead_raw = tf.cast(tf.round(x[:, self.bead_col]), tf.int32)
        # With ALA filtered out, gate ids are expected in 1..18.
        tf.debugging.assert_greater_equal(gate_raw, 1, message="Gate values below 1 found.")
        tf.debugging.assert_less_equal(gate_raw, self.num_experts, message="Gate values above num_experts found.")
        tf.debugging.assert_greater_equal(bead_raw, 0, message="Negative bead ids found.")
        tf.debugging.assert_less_equal(bead_raw, 3, message="Bead ids above 3 found.")

    def train_step(self, data):
        x, y, sw = tf.keras.utils.unpack_x_y_sample_weight(data)


# (A) Tripwires on inputs (optional but useful)
        tf.debugging.assert_all_finite(tf.cast(x, tf.float32), "x has NaN/Inf")
        tf.debugging.assert_all_finite(tf.cast(y, tf.float32), "y has NaN/Inf")
        tf.debugging.assert_all_finite(tf.cast(sw, tf.float32), "sample_weight/mask has NaN/Inf")
        self._routing_assertions(x)



        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # (B) Tripwire: model output
            tf.debugging.assert_all_finite(y_pred, "y_pred has NaN/Inf (forward pass)")

            coord_loss = self.loss3(y, y_pred, sw)
            self.coord_tracker.update_state(coord_loss)

            tf.debugging.assert_all_finite(coord_loss, "coord_loss has NaN/Inf")



            bond_loss = tf.constant(0.0, dtype=tf.float32)
            if self.use_bond_loss:
                bond_loss = bond_mse_from_tables(
                    x=x, y_true=y, y_pred=y_pred, mask=sw,
                    bond_pairs_table=self.bond_pairs_table,
                    bond_valid_table=self.bond_valid_table,
                    template_of_res_bead=self.template_of_res_bead,
                    n_atoms=self.n_atoms,
                    cont_len=self.cont_len,
                    gate_offset=self.gate_offset,
                    num_experts=self.num_experts,
                    bead_col=self.bead_col,
                )

                tf.debugging.assert_all_finite(bond_loss, "bond_loss has NaN/Inf")


            angle_loss = tf.constant(0.0, dtype=tf.float32)
            if self.use_angle_loss:
                angle_loss = angle_mse_from_tables_stable(
                    x=x, y_true=y, y_pred=y_pred, mask=sw,
                    angle_triplets_table=self.angle_triplets_table,
                    angle_valid_table=self.angle_valid_table,
                    n_atoms=self.n_atoms,
                    cont_len=self.cont_len,
                    gate_offset=self.gate_offset,
                    num_experts=self.num_experts,
                    bead_col=self.bead_col,
                    max_beads=4,
                    min_len=2e-2,  # try 0.02 then 0.03 then 0.05
                )











                tf.debugging.assert_all_finite(angle_loss, "angle_loss has NaN/Inf")


            #reg_loss = tf.add_n(self.losses) if self.losses else 0.0
            #loss = coord_loss + self.bond_weight * bond_loss + self.angle_weight * angle_loss  #+ reg_loss

            loss = coord_loss + self.bond_weight * bond_loss + self.angle_weight * angle_loss

            # (C) Tripwire: total loss
            tf.debugging.assert_all_finite(loss, "total loss has NaN/Inf")



        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))


        # (D) Tripwire: gradients
        for g in grads:
            if g is not None:
                tf.debugging.assert_all_finite(g, "a gradient has NaN/Inf")


        rmse = self.rmse3(y, y_pred, sw)


        tf.debugging.assert_all_finite(rmse, "rmse metric has NaN/Inf")

        self.loss_tracker.update_state(loss)
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

    def test_step(self, data):
        x, y, sw = tf.keras.utils.unpack_x_y_sample_weight(data)
        self._routing_assertions(x)
        y_pred = self(x, training=False)

        coord_loss = self.loss3(y, y_pred, sw)

        bond_loss = tf.constant(0.0, dtype=tf.float32)
        if self.use_bond_loss:
            bond_loss = bond_mse_from_tables(
                x=x, y_true=y, y_pred=y_pred, mask=sw,
                bond_pairs_table=self.bond_pairs_table,
                bond_valid_table=self.bond_valid_table,
                template_of_res_bead=self.template_of_res_bead,
                n_atoms=self.n_atoms,
                cont_len=self.cont_len,
                gate_offset=self.gate_offset,
                num_experts=self.num_experts,
                bead_col=self.bead_col,
            )

        angle_loss = tf.constant(0.0, dtype=tf.float32)
        if self.use_angle_loss:
            angle_loss = angle_mse_from_tables_stable(
                x=x, y_true=y, y_pred=y_pred, mask=sw,
                angle_triplets_table=self.angle_triplets_table,
                angle_valid_table=self.angle_valid_table,
                n_atoms=self.n_atoms,
                cont_len=self.cont_len,
                gate_offset=self.gate_offset,
                num_experts=self.num_experts,
                bead_col=self.bead_col,
                max_beads=4,
                min_len=2e-2,  # try 0.02 then 0.03 then 0.05
            )








        reg_loss = tf.add_n(self.losses) if self.losses else 0.0
        loss = coord_loss + self.bond_weight * bond_loss + self.angle_weight * angle_loss #+ reg_loss

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
            "angle_mse": self.angle_tracker.result()
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
    input_dim=38,
    cont_len=36,
    y_dim=15,
    num_experts=18,
    other_cat_vocab=4,
    other_cat_emb_dim=8,
    gate_offset=1,
    gate_emb_dim=8,
    trunk_width=512,
    trunk_blocks=8,          # use 6 or 8
    fusion_width=512,
    latent_dim=512,
    expert_hidden=(512, 256),# or (256, 128)
    dropout=0.1,
    keep_mask_input=False,   # set True only if you still want mask as a model input
):
    # Inputs
    x_in = layers.Input(shape=(input_dim,), dtype=tf.float32, name="x")
    if keep_mask_input:
        mask_in = layers.Input(shape=(y_dim,), dtype=tf.float32, name="mask")

    # Slice continuous + categoricals
    cont = layers.Lambda(lambda t: t[:, :cont_len], name="cont_slice")(x_in)            # (B,25)
    other_cat = layers.Lambda(lambda t: t[:, cont_len], name="cat25_slice")(x_in)       # (B,)
    gate_cat  = layers.Lambda(lambda t: t[:, cont_len + 1], name="gate26_slice")(x_in)  # (B,)

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

def materialize_filtered(src, indices, out_path, chunk_size=200000):
    indices = np.asarray(indices, dtype=np.int64)
    out_shape = (indices.shape[0], src.shape[1])
    out = open_memmap(out_path, mode="w+", dtype=np.float32, shape=out_shape)
    for s in range(0, indices.shape[0], chunk_size):
        e = min(s + chunk_size, indices.shape[0])
        out[s:e] = np.asarray(src[indices[s:e]], dtype=np.float32)
    del out
    return np.load(out_path, mmap_mode="r")




X = np.load("local_Frame2/COMBINED_SIDECHAIN/train_features_allPDBs.npy", mmap_mode="r")
Y = np.load("local_Frame2/COMBINED_SIDECHAIN/train_targets_allPDBs.npy",  mmap_mode="r")
MASK = np.load("local_Frame2/COMBINED_SIDECHAIN/train_masks_allPDBs.npy", mmap_mode="r")

X_test = np.load("local_Frame2/COMBINED_SIDECHAIN/test_features_allPDBs.npy", mmap_mode="r")
Y_test = np.load("local_Frame2/COMBINED_SIDECHAIN/test_targets_allPDBs.npy",  mmap_mode="r")
MASK_test = np.load("local_Frame2/COMBINED_SIDECHAIN/test_masks_allPDBs.npy", mmap_mode="r")



ALA_ID = 0  # <-- CHANGE THIS to whatever your alanine id is in X[:, 26]
FEATURE_DIM = 38
CONT_LEN = 36
BEAD_COL = 36
GATE_COL = 37

def summarize_filter(X, name="train"):
    if X.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected X with {FEATURE_DIM} columns, got {X.shape}")
    gate = np.rint(X[:, GATE_COL]).astype(np.int32)   # rounding avoids 17.999 -> 17 surprises
    keep = gate != ALA_ID

    n = gate.shape[0]
    kept = int(keep.sum())
    removed = n - kept

    print(f"[{name}] total: {n:,}")
    print(f"[{name}] keep : {kept:,} ({kept/n:.2%})")
    print(f"[{name}] drop : {removed:,} ({removed/n:.2%})")

    # sanity checks
    if kept == 0:
        raise RuntimeError("Filter kept 0 rows -> ALA_ID is probably wrong or gate column is wrong.")
    if removed == 0:
        print("Warning: removed 0 rows -> either no Ala present or ALA_ID is wrong.")

    # check a few removed gates are exactly ALA_ID
    if removed > 0:
        removed_ids = np.unique(gate[~keep])
        print(f"[{name}] unique gate ids removed: {removed_ids[:20]}{'...' if removed_ids.size>20 else ''}")

    # optional: show distribution of gate ids
    vals, cnts = np.unique(gate, return_counts=True)
    top = np.argsort(-cnts)[:10]
    print(f"[{name}] top gate ids: {list(zip(vals[top], cnts[top]))}")
    return keep




keep_train = summarize_filter(X, "train")
train_idx = np.where(keep_train)[0].astype(np.int64)
del keep_train

keep_test = summarize_filter(X_test, "test")
test_idx = np.where(keep_test)[0].astype(np.int64)
del keep_test

print("train kept rows:", train_idx.shape[0], "/", X.shape[0])
print("test kept rows :", test_idx.shape[0], "/", X_test.shape[0])
CACHE_DIR = "local_Frame2/COMBINED_SIDECHAIN/_filtered_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
REBUILD_FILTERED = int(os.getenv("REBUILD_FILTERED", "0")) == 1
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2048"))
print("batch_size:", BATCH_SIZE)

train_x_path = os.path.join(CACHE_DIR, "train_X_filtered.npy")
train_y_path = os.path.join(CACHE_DIR, "train_Y_filtered.npy")
train_m_path = os.path.join(CACHE_DIR, "train_M_filtered.npy")
test_x_path = os.path.join(CACHE_DIR, "test_X_filtered.npy")
test_y_path = os.path.join(CACHE_DIR, "test_Y_filtered.npy")
test_m_path = os.path.join(CACHE_DIR, "test_M_filtered.npy")

if REBUILD_FILTERED or not (os.path.exists(train_x_path) and os.path.exists(train_y_path) and os.path.exists(train_m_path)):
    print("Building filtered train cache...")
    X_f = materialize_filtered(X, train_idx, train_x_path)
    Y_f = materialize_filtered(Y, train_idx, train_y_path)
    M_f = materialize_filtered(MASK, train_idx, train_m_path)
else:
    X_f = np.load(train_x_path, mmap_mode="r")
    Y_f = np.load(train_y_path, mmap_mode="r")
    M_f = np.load(train_m_path, mmap_mode="r")

if REBUILD_FILTERED or not (os.path.exists(test_x_path) and os.path.exists(test_y_path) and os.path.exists(test_m_path)):
    print("Building filtered test cache...")
    X_test_f = materialize_filtered(X_test, test_idx, test_x_path)
    Y_test_f = materialize_filtered(Y_test, test_idx, test_y_path)
    M_test_f = materialize_filtered(MASK_test, test_idx, test_m_path)
else:
    X_test_f = np.load(test_x_path, mmap_mode="r")
    Y_test_f = np.load(test_y_path, mmap_mode="r")
    M_test_f = np.load(test_m_path, mmap_mode="r")

print('the shape of X train is:', X_f.shape)
print('the shape of Y train is as follows: ', Y_f.shape)
print('the shape of Mask is as follows:   ', M_f.shape)


#print( X[0,:])
#print( Y[0,:])
#print(MASK[0,:])


#==============================================================================================================================================
# =========================
# 7) Train + checkpoints
# =========================
strategy = tf.distribute.MirroredStrategy()
print("Replicas:", strategy.num_replicas_in_sync)
bond_tables = (bond_pairs_table, bond_valid_table, template_of_res_bead)


angle_trip_tf, angle_val_tf = angle_tables  # from build_angle_tables(...)
angle_tables = (angle_trip_tf, angle_val_tf)

ckpts = glob.glob("ckpt_epoch_EXPERT_M14_*.keras")
assert ckpts, "No ckpt_epoch_EXPERT_M14_*.keras files found."

def epoch_from_name(p):
    m = re.search(r"_([0-9]+)\.keras$", p)
    return int(m.group(1)) if m else -1

latest_ckpt = max(ckpts, key=epoch_from_name)
last_epoch  = epoch_from_name(latest_ckpt)  # e.g. 37 means you completed epoch 37

print("Latest checkpoint:", latest_ckpt)
print("Last completed epoch:", last_epoch)


with strategy.scope():


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




     #   print(model.weights)
    init_lr = 1e-4  # float

    opt = tf.keras.optimizers.Adam(
        learning_rate=init_lr,   # <-- float ONLY
        clipnorm=1.0
    )

    model = MaskAbsorbingModel(
        inputs=base.inputs,
        outputs=base.outputs,
        loss3=atom_mse_norm_by_mask(n_atoms=5),
        rmse3=atom_rmse_norm_by_mask(n_atoms=5),

        bond_weight= 2.0,    
        bond_tables=bond_tables,

        angle_weight= 1,     # start small: 0.005–0.05
        angle_tables=angle_tables,

        bead_col=36,
        n_atoms=5,
        cont_len=CONT_LEN,
        gate_offset=1,
        num_experts=18
    )
    model.compile(optimizer=opt)



#model.load_weights(latest_ckpt)
#print("Loaded weights from:", latest_ckpt)

print(model.summary())

#print(model.weights)
#

ckpt_all = tf.keras.callbacks.ModelCheckpoint(
    filepath="ckpt_epoch_EXPERT_M24_{epoch:02d}.keras",
    save_weights_only=False,
    save_freq="epoch",
    verbose=1,
)

ckpt_best = tf.keras.callbacks.ModelCheckpoint(
    filepath="EXPERT_M24_best.weights.h5",
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)






model.fit(
    X_f, Y_f,
    sample_weight=M_f,
    validation_data=(X_test_f, Y_test_f, M_test_f),
    epochs=20000,
    batch_size=BATCH_SIZE,
    
 #   steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[ ckpt_all, ckpt_best]
)






#hard = model.get_layer("hard_route")
#print("num experts:", len(hard.experts))
#print("expert[0] summary:")
#print(hard.experts[0].summary())
#print("expert[19] summary:")
#print(hard.experts[19].summary())
