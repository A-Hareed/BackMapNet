import numpy as np

# 1. Collect all (phi, psi) from your PDB dataset (in degrees or radians)
phis = np.concatenate(all_phi_lists)   # shape [N_total]
psis = np.concatenate(all_psi_lists]   # shape [N_total]

# 2. Compute a 2D histogram
n_bins = 72  # e.g. 5° bins over [-180,180) 
edges = np.linspace(-np.pi, np.pi, n_bins+1)
H, xedges, yedges = np.histogram2d(phis, psis, bins=[edges, edges], density=True)

# 3. Convert to log‐prob and save
logp = np.log(H + 1e-6)  # add small ε to avoid log(0)
np.savez("rama_prior.npz", logp=logp.astype(np.float32),
                             phi_edges=xedges.astype(np.float32),
                             psi_edges=yedges.astype(np.float32))

array_AA = (batch_LAB - custom_min) / custom_range
np.sum( ((batch_LAB - custom_min) - array_AA*custom_range)))

(2180000, 32, 4, 3)



import numpy as np

def dihedral(p0, p1, p2, p3):
    """
    Compute dihedral angle (in degrees) for four points p0→p1→p2→p3.
    p#: arrays of shape (..., 3).
    Returns an array of shape (...) of signed angles in degrees.
    """
    # Bond vectors
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    # Normalize b1 for projection
    b1_norm = b1 / np.linalg.norm(b1, axis=-1)[..., None]

    # Orthogonal components
    v = b0 - np.sum(b0 * b1_norm, axis=-1)[..., None] * b1_norm
    w = b2 - np.sum(b2 * b1_norm, axis=-1)[..., None] * b1_norm

    # Compute x = v·w, y = (b1_norm×v)·w
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1_norm, v) * w, axis=-1)

    return np.degrees(np.arctan2(y, x))


# coords: your array of shape (batch, 32, 4, 3)
coords = YOUR_ARRAY  

# Grab N, CA, C atom coordinate slices:
N  = array_AA[:, :, 0, :]   # shape (batch, 32, 3)
CA = array_AA[:, :, 1, :]
C  = array_AA[:, :, 2, :]

# φ (phi) for residues 1…31 uses C(i-1), N(i), CA(i), C(i)
phi = dihedral(
    C[:, :-1, :],     # C(i-1)
    N[:, 1:,  :],     # N(i)
    CA[:, 1:, :],     # CA(i)
    C[:, 1:,  :]      # C(i)
)
# phi shape → (batch, 31)

# ψ (psi) for residues 0…30 uses N(i), CA(i), C(i), N(i+1)
psi = dihedral(
    N[:, :-1, :],     # N(i)
    CA[:, :-1, :],    # CA(i)
    C[:, :-1,  :],    # C(i)
    N[:, 1:,   :]     # N(i+1)
)
# psi shape → (batch, 31)

# If you want φ and ψ both as (batch, 32), you can pad the ends with NaN:
phi_full = np.full((coords.shape[0], coords.shape[1]), np.nan)
phi_full[:, 1:] = phi

psi_full = np.full((coords.shape[0], coords.shape[1]), np.nan)
psi_full[:, :-1] = psi

# Now phi_full and psi_full are both (batch, 32), with φ[0]=NaN, ψ[31]=NaN.

 
