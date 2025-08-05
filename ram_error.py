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

 
