import os
import numpy as np
import tensorflow as tf

from torsion_loss4 import torsion_mse_loss_fast


# Runtime-configured globals populated by configure_rama_prior().
_PRIOR_GRID = None
_BINS = None
_BIN_WIDTH_TF = None
_T_TF = None

_PI = tf.constant(np.pi, dtype=tf.float32)
_ONE_F = tf.constant(1.0, dtype=tf.float32)


def resolve_prior_path(prior_path=None):
    if prior_path and os.path.exists(prior_path):
        return os.path.abspath(prior_path)

    env_path = os.environ.get("RAMA_PRIOR_PATH", "").strip()
    if env_path and os.path.exists(env_path):
        return os.path.abspath(env_path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_default = os.path.join(script_dir, "..", "weights", "RamachandranEval_prors.npy")
    if os.path.exists(repo_default):
        return os.path.abspath(repo_default)

    legacy = os.path.abspath("RamachandranEval_prors.npy")
    if os.path.exists(legacy):
        return legacy

    raise FileNotFoundError(
        "Ramachandran prior not found. Expected one of:\n"
        f"  provided path={prior_path or '<none>'}\n"
        f"  RAMA_PRIOR_PATH={env_path or '<unset>'}\n"
        f"  {repo_default}\n"
        f"  {legacy}"
    )


def configure_rama_prior(prior_path=None, percentile=18):
    global _PRIOR_GRID, _BINS, _BIN_WIDTH_TF, _T_TF

    path = resolve_prior_path(prior_path)
    neglog = np.load(path).T.astype(np.float32)

    _PRIOR_GRID = tf.constant(neglog, dtype=tf.float32)
    _BINS = int(neglog.shape[0])
    _BIN_WIDTH_TF = tf.constant(360.0 / float(_BINS), dtype=tf.float32)
    _T_TF = tf.constant(float(np.percentile(neglog.ravel(), percentile)), dtype=tf.float32)

    return path


def _check_prior():
    if _PRIOR_GRID is None or _BINS is None or _BIN_WIDTH_TF is None or _T_TF is None:
        raise RuntimeError("Ramachandran prior is not configured. Call configure_rama_prior(...) first.")


def combined_torsion_loss(y_true, y_pred):
    coords_pred = y_pred[:, :384]
    ranges_raw = y_pred[:, 384:]
    ranges = tf.expand_dims(ranges_raw, axis=1)

    return torsion_mse_loss_fast(
        norm_coords=coords_pred,
        true_norm_coords=y_true,
        ranges=ranges,
    )


def normalized_coord_mse(y_true, y_pred):
    coords_pred = y_pred[:, :384]
    coords_true = y_true[:, :384]
    return tf.reduce_mean(tf.square(coords_pred - coords_true))


def _tf_sample_bilinear(grid, phi_deg, psi_deg, bins, bin_width_tf):
    grid = tf.cast(grid, tf.float32)
    phi_deg = tf.cast(phi_deg, tf.float32)
    psi_deg = tf.cast(psi_deg, tf.float32)

    x = phi_deg / bin_width_tf
    y = psi_deg / bin_width_tf

    x0 = tf.floor(x)
    y0 = tf.floor(y)
    x_frac = x - x0
    y_frac = y - y0

    x0_i = tf.cast(tf.math.floormod(tf.cast(x0, tf.int32), bins), tf.int32)
    y0_i = tf.cast(tf.math.floormod(tf.cast(y0, tf.int32), bins), tf.int32)
    x1_i = tf.cast(tf.math.floormod(x0_i + 1, bins), tf.int32)
    y1_i = tf.cast(tf.math.floormod(y0_i + 1, bins), tf.int32)

    def gather_vals(ix, iy):
        ix_flat = tf.reshape(ix, [-1])
        iy_flat = tf.reshape(iy, [-1])
        idx = tf.stack([ix_flat, iy_flat], axis=1)
        vals = tf.gather_nd(grid, idx)
        return tf.reshape(vals, tf.shape(ix))

    q11 = gather_vals(x0_i, y0_i)
    q21 = gather_vals(x1_i, y0_i)
    q12 = gather_vals(x0_i, y1_i)
    q22 = gather_vals(x1_i, y1_i)

    wx = tf.cast(x_frac, tf.float32)
    wy = tf.cast(y_frac, tf.float32)

    top = q11 * (_ONE_F - wx) + q21 * wx
    bottom = q12 * (_ONE_F - wx) + q22 * wx
    return top * (_ONE_F - wy) + bottom * wy


@tf.function
def rama_penalty(y_true, y_pred, use_squared_hinge=False, return_fraction=False):
    _check_prior()

    coords_pred = y_pred[:, :384]
    ranges_raw = y_pred[:, 384:]
    ranges = tf.expand_dims(ranges_raw, axis=1)

    phi_psi_rad = torsion_mse_loss_fast(norm_coords=coords_pred, ranges=ranges)
    phi_rad = phi_psi_rad[:, :31]
    psi_rad = phi_psi_rad[:, 31:]

    valid = tf.math.is_finite(phi_rad) & tf.math.is_finite(psi_rad)

    phi_deg = tf.math.floormod((phi_rad * 180.0 / _PI) + 180.0, 360.0)
    psi_deg = tf.math.floormod((psi_rad * 180.0 / _PI) + 180.0, 360.0)

    sampled_neglog = _tf_sample_bilinear(_PRIOR_GRID, phi_deg, psi_deg, _BINS, _BIN_WIDTH_TF)
    sampled_neglog = tf.where(valid, sampled_neglog, tf.zeros_like(sampled_neglog))

    hinge_vals = tf.nn.relu(sampled_neglog - _T_TF)
    if use_squared_hinge:
        hinge_vals = tf.square(hinge_vals)

    counts = tf.reduce_sum(tf.cast(valid, tf.float32), axis=1)
    sums = tf.reduce_sum(hinge_vals, axis=1)
    rama_per_example = sums / (counts + 1e-6)

    is_forbidden = tf.cast(sampled_neglog > _T_TF, tf.float32)
    fraction_forbidden = tf.reduce_sum(is_forbidden * tf.cast(valid, tf.float32), axis=1) / (counts + 1e-6)

    if return_fraction:
        return rama_per_example, fraction_forbidden
    return rama_per_example


def combined_coord_and_torsion_loss(y_true, y_pred):
    scale_mse = 0.01000
    mse = normalized_coord_mse(y_true, y_pred)
    rama_loss = rama_penalty(y_true, y_pred)
    return mse + scale_mse * rama_loss


class CoordRMSE(tf.keras.metrics.RootMeanSquaredError):
    def update_state(self, y_true, y_pred, sample_weight=None):
        coords_pred = y_pred[:, :384]
        return super().update_state(y_true[:, :384], coords_pred, sample_weight)


def phi_metric(y_true, y_pred):
    phi_loss = combined_torsion_loss(y_true, y_pred)
    return tf.reduce_mean(phi_loss[:, :31])


def psi_metric(y_true, y_pred):
    psi_loss = combined_torsion_loss(y_true, y_pred)
    return tf.reduce_mean(psi_loss[:, 31:])


class FractionForbidden(tf.keras.metrics.Metric):
    def __init__(self, prior_grid, bin_width, threshold, bins, name="fraction_forbidden", dtype=tf.float32, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.prior_grid = tf.cast(prior_grid, tf.float32)
        self.bin_width = tf.cast(bin_width, tf.float32)
        self.threshold = tf.cast(threshold, tf.float32)
        self.bins = int(bins)
        self.total = self.add_weight(name="total", initializer="zeros", dtype=dtype)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=dtype)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

    # Compatibility with older code paths that still call reset_states.
    def reset_states(self):
        self.reset_state()

    def _sample_bilinear(self, phi_deg, psi_deg):
        x = phi_deg / self.bin_width
        y = psi_deg / self.bin_width

        x0 = tf.floor(x)
        y0 = tf.floor(y)
        x_frac = x - x0
        y_frac = y - y0

        x0_i = tf.cast(tf.math.floormod(tf.cast(x0, tf.int32), self.bins), tf.int32)
        y0_i = tf.cast(tf.math.floormod(tf.cast(y0, tf.int32), self.bins), tf.int32)
        x1_i = tf.cast(tf.math.floormod(x0_i + 1, self.bins), tf.int32)
        y1_i = tf.cast(tf.math.floormod(y0_i + 1, self.bins), tf.int32)

        def gather_vals(ix, iy):
            ix_flat = tf.reshape(ix, [-1])
            iy_flat = tf.reshape(iy, [-1])
            idx = tf.stack([ix_flat, iy_flat], axis=1)
            vals = tf.gather_nd(self.prior_grid, idx)
            return tf.reshape(vals, tf.shape(ix))

        q11 = gather_vals(x0_i, y0_i)
        q21 = gather_vals(x1_i, y0_i)
        q12 = gather_vals(x0_i, y1_i)
        q22 = gather_vals(x1_i, y1_i)

        wx = tf.cast(x_frac, tf.float32)
        wy = tf.cast(y_frac, tf.float32)

        top = q11 * (1.0 - wx) + q21 * wx
        bottom = q12 * (1.0 - wx) + q22 * wx
        return top * (1.0 - wy) + bottom * wy

    def update_state(self, y_true, y_pred, sample_weight=None):
        coords_pred = y_pred[:, :384]
        ranges_raw = y_pred[:, 384:]
        ranges = tf.expand_dims(ranges_raw, axis=1)

        phi_psi_rad = torsion_mse_loss_fast(norm_coords=coords_pred, ranges=ranges)
        phi_rad = phi_psi_rad[:, :31]
        psi_rad = phi_psi_rad[:, 31:]

        valid = tf.math.is_finite(phi_rad) & tf.math.is_finite(psi_rad)
        phi_deg = tf.math.floormod((phi_rad * 180.0 / _PI) + 180.0, 360.0)
        psi_deg = tf.math.floormod((psi_rad * 180.0 / _PI) + 180.0, 360.0)

        sampled_neglog = self._sample_bilinear(phi_deg, psi_deg)
        sampled_neglog = tf.where(valid, sampled_neglog, tf.zeros_like(sampled_neglog))

        is_forbidden = tf.cast(sampled_neglog > self.threshold, tf.float32)
        forbidden_counts = tf.reduce_sum(is_forbidden * tf.cast(valid, tf.float32), axis=1)
        counts = tf.reduce_sum(tf.cast(valid, tf.float32), axis=1)
        fraction_per_example = forbidden_counts / (counts + 1e-6)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            if tf.rank(sample_weight) == 0:
                sample_weight = tf.fill(tf.shape(fraction_per_example), sample_weight)
            weighted_sum = tf.reduce_sum(fraction_per_example * sample_weight)
            denom = tf.reduce_sum(sample_weight)
            batch_mean = weighted_sum / (denom + 1e-6)
        else:
            batch_mean = tf.reduce_mean(fraction_per_example)

        self.total.assign_add(batch_mean)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / (self.count + 1e-12)


def make_custom_objects(prior_path=None, percentile=18):
    configure_rama_prior(prior_path=prior_path, percentile=percentile)

    frac_metric = FractionForbidden(
        prior_grid=_PRIOR_GRID,
        bin_width=_BIN_WIDTH_TF,
        threshold=_T_TF,
        bins=_BINS,
    )

    return {
        "combined_torsion_loss": combined_torsion_loss,
        "normalized_coord_mse": normalized_coord_mse,
        "rama_penalty": rama_penalty,
        "combined_coord_and_torsion_loss": combined_coord_and_torsion_loss,
        "CoordRMSE": CoordRMSE,
        "FractionForbidden": FractionForbidden,
        "phi_metric": phi_metric,
        "psi_metric": psi_metric,
        "frac_metric": frac_metric,
    }
