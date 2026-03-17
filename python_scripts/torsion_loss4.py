import numpy as np
import tensorflow as tf

# Constants
RESIDUE_COUNT = 32
ATOM_PER_RES  = 4   # N, CA, C, O
QUAD_COUNT    = (RESIDUE_COUNT - 1) * 2  # 31 φ + 31 ψ = 62

# 1) Indices of backbone atoms (N=0, CA=1, C=2) in each residue block → [32,3]
atom_indices = tf.constant(np.stack([
    np.arange(0, RESIDUE_COUNT*ATOM_PER_RES, ATOM_PER_RES),
    np.arange(1, RESIDUE_COUNT*ATOM_PER_RES, ATOM_PER_RES),
    np.arange(2, RESIDUE_COUNT*ATOM_PER_RES, ATOM_PER_RES),
], axis=1), dtype=tf.int32)

# 2) Build quartet index array against the *full* 128‑atom buffer → [62,4]
phi_idxs = np.stack([
    atom_indices[:-1,2],  # C_{i-1}
    atom_indices[1: ,0],  # N_i
    atom_indices[1: ,1],  # CA_i
    atom_indices[1: ,2],  # C_i
], axis=1)  # [31,4]
psi_idxs = np.stack([
    atom_indices[:-1,0],  # N_i
    atom_indices[:-1,1],  # CA_i
    atom_indices[:-1,2],  # C_i
    atom_indices[1: ,0],  # N_{i+1}
], axis=1)  # [31,4]
quad_indices = tf.constant(
    np.concatenate([phi_idxs, psi_idxs], axis=0),
    dtype=tf.int32
)  # [62,4]

@tf.function
def torsion_mse_loss_fast(norm_coords, ranges):
    B = tf.shape(norm_coords)[0]

    # 1) Reshape & descale → [B,128,3]
    coords_p = tf.reshape(norm_coords,     [B, RESIDUE_COUNT*ATOM_PER_RES, 3]) * ranges


    # 2) Gather each quartet directly from the 128‑atom buffer → [B,62,4,3]
    quads_p = tf.gather(coords_p, quad_indices, axis=1)


    # 3) Compute signed torsions
    def compute_torsions(quads):
        A, Bv, C, D = tf.unstack(quads, axis=2)
        b1 = Bv - A
        b2 = C   - Bv
        b3 = D   - C
        n1 = tf.linalg.cross(b1, b2)
        n2 = tf.linalg.cross(b2, b3)
        x  = tf.reduce_sum(n1 * n2, axis=-1)                    # [B,62]
        y  = tf.norm(b2, axis=-1) * tf.reduce_sum(n1 * b3, axis=-1)
        return tf.atan2(y, x)                                   # [B,62]

    theta_p = compute_torsions(quads_p)



    return theta_p 