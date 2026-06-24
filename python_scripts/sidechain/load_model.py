import argparse
import os
import json
import numpy as np
import tensorflow as tf


def _install_dense_quantization_config_compat_patch():
    """
    Newer Keras can save Dense configs with quantization_config=None.
    Older Keras builds fail on that unknown kwarg even though it carries no data.
    """

    dense_classes = [tf.keras.layers.Dense]
    try:
        import keras

        dense_classes.append(keras.layers.Dense)
    except Exception:
        pass
    try:
        from keras.src.layers.core.dense import Dense as KerasSrcDense

        dense_classes.append(KerasSrcDense)
    except Exception:
        pass

    seen = set()
    for dense_cls in dense_classes:
        if id(dense_cls) in seen:
            continue
        seen.add(id(dense_cls))
        if getattr(dense_cls, "_backmapnet_quant_config_patch", False):
            continue

        original_from_config = dense_cls.from_config

        @classmethod
        def patched_from_config(cls, config, _original=original_from_config):
            if (
                isinstance(config, dict)
                and "quantization_config" in config
                and config["quantization_config"] is None
            ):
                config = dict(config)
                config.pop("quantization_config", None)
            return _original(config)

        dense_cls.from_config = patched_from_config
        dense_cls._backmapnet_quant_config_patch = True


def _load_optional_custom_objects():
    """
    Portable .keras models may reference custom classes from training code.
    Try to import them if available, but keep fallback empty so plain models load.
    """
    custom_objects = {}
    class_names = (
        "SliceColumns",
        "RoundClipIndex",
        "GeometryAwareMaskedMSE",
        "GeometryAwareCoordMSE",
        "GeometryAwareCoordRMSE",
        "CoordMainBondAuxModel",
    )

    for module_name in ("multi_expert_model_40", "multi_expert_model_38"):
        try:
            module = __import__(module_name, fromlist=class_names)
            for name in class_names:
                obj = getattr(module, name, None)
                if obj is not None and name not in custom_objects:
                    custom_objects[name] = obj
        except Exception:
            # No-op: many exported models do not need these at inference time.
            pass
    return custom_objects


def _resolve_geometry_eval_objects(custom_objects):
    required = (
        "GeometryAwareMaskedMSE",
        "GeometryAwareCoordMSE",
        "GeometryAwareCoordRMSE",
    )
    missing = [k for k in required if k not in custom_objects]
    if missing:
        raise ImportError(
            "Missing required custom geometry objects for evaluation: "
            + ", ".join(missing)
            + ". Ensure multi_expert_model_38.py or multi_expert_model_40.py is importable."
        )
    return {k: custom_objects[k] for k in required}


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.abspath(
        os.path.join(script_dir, "..", "..", "weights", "ckpt_epoch_SINGLE_BASE_37_147.keras")
    )

    parser = argparse.ArgumentParser(
        description="Load side-chain portable .keras model and run evaluate/predict."
    )

    # Positional pipeline-style inputs (preferred in this framework)
    parser.add_argument("feature_file", nargs="?", help="Feature file (.npy/.npz)")
    parser.add_argument("target_file", nargs="?", help="Target file (.npy/.npz)")
    parser.add_argument("mask_file", nargs="?", help="Mask file (.npy/.npz)")
    parser.add_argument("cluster", nargs="?", help="Cluster id for output naming")

    # Optional explicit inputs for ad-hoc usage
    parser.add_argument("--x", default="", help="Feature file (.npy/.npz)")
    parser.add_argument("--y", default="", help="Target file (.npy/.npz)")
    parser.add_argument("--m", default="", help="Mask/sample-weight file (.npy/.npz)")

    parser.add_argument(
        "--model",
        default=default_model,
        help="Path to side-chain .keras model file",
    )
    parser.add_argument(
        "--model40-manifest",
        default="",
        help=(
            "Optional manifest JSON from multi_expert_model_40.py "
            "(contains residue/bead scaling hyperparameters). "
            "Can also be set via SIDECHAIN_MODEL40_MANIFEST."
        ),
    )
    parser.add_argument("--out", default=None, help="Prediction output .npy filename")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation step")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size")

    return parser.parse_args()


def load_arr(path, mmap_npy=True, require_arr_key=False):
    if path.endswith(".npz"):
        arr = np.load(path)
        if require_arr_key:
            if "arr" not in arr:
                raise KeyError(
                    f"NPZ file {path} is missing required key 'arr'. "
                    f"Available keys: {list(arr.keys())}"
                )
            return np.asarray(arr["arr"], dtype=np.float32)
        if "arr" in arr:
            return np.asarray(arr["arr"], dtype=np.float32)
        first_key = list(arr.keys())[0]
        return np.asarray(arr[first_key], dtype=np.float32)
    if mmap_npy:
        arr = np.load(path, mmap_mode="r")
        if arr.dtype != np.float32:
            return np.asarray(arr, dtype=np.float32)
        return arr
    return np.asarray(np.load(path), dtype=np.float32)


def resolve_inputs(args):
    x_path = args.feature_file or args.x
    y_path = args.target_file or args.y
    m_path = args.mask_file or args.m

    if not x_path:
        raise ValueError("Missing feature file. Provide positional feature_file or --x.")
    if not args.no_eval and (not y_path or not m_path):
        raise ValueError("Evaluation mode requires target/mask. Provide positional target/mask or --y/--m.")

    return x_path, y_path, m_path


def pick_output_name(args):
    if args.out:
        return args.out
    if args.cluster:
        return f"expert_{args.cluster}_Yhat.npy"
    return "expert_Yhat.npy"


def adapt_feature_width(x, expected_width):
    if x.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape {x.shape}")

    if x.shape[1] == expected_width:
        return x

    # Legacy side-chain feature tensors can be width 38 while some portable models expect 36.
    if x.shape[1] == 38 and expected_width == 36:
        keep = [i for i in range(38) if i not in (36, 37)]
        x = x[:, keep]
        print("Adjusted feature width 38->36 by dropping columns 36 and 37.")
        return x

    raise ValueError(
        f"Feature width mismatch: model expects {expected_width}, input has {x.shape[1]}"
    )


def _candidate_model40_manifest_paths(model_path, explicit_manifest=""):
    candidates = []
    if explicit_manifest:
        candidates.append(explicit_manifest)

    env_manifest = os.getenv("SIDECHAIN_MODEL40_MANIFEST", "").strip()
    if env_manifest:
        candidates.append(env_manifest)

    model_path = os.path.abspath(os.path.expanduser(model_path))
    if model_path.endswith("_portable.keras"):
        candidates.append(model_path.replace("_portable.keras", "_manifest.json"))
    if model_path.endswith(".keras"):
        candidates.append(model_path[:-len(".keras")] + "_manifest.json")

    deduped = []
    for path in candidates:
        abs_path = os.path.abspath(os.path.expanduser(path))
        if abs_path not in deduped:
            deduped.append(abs_path)
    return deduped


def _load_model40_scaling_config(model_path, explicit_manifest=""):
    required = ("residue_col", "bead_col", "residue_offset", "num_res_types", "max_beads")
    for manifest_path in _candidate_model40_manifest_paths(
        model_path=model_path,
        explicit_manifest=explicit_manifest,
    ):
        if not os.path.isfile(manifest_path):
            continue
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        hp = manifest.get("hyperparameters", {})
        missing = [k for k in required if k not in hp]
        if missing:
            raise KeyError(
                f"Manifest missing keys {missing}: {manifest_path}"
            )
        cfg = {k: int(hp[k]) for k in required}
        cfg["manifest_path"] = manifest_path
        return cfg
    return None


def _is_likely_model40(model_path):
    name = os.path.basename(model_path).lower()
    return ("_bond_" in name) or ("model_40" in name) or ("40_bond" in name)


def maybe_apply_model40_input_scaling(x, expected_width, model_path, explicit_manifest=""):
    if int(expected_width) != 38:
        return x

    cfg = _load_model40_scaling_config(model_path=model_path, explicit_manifest=explicit_manifest)

    if cfg is None:
        if _is_likely_model40(model_path):
            raise FileNotFoundError(
                "Model appears to be model_40 but no manifest was found for input scaling. "
                "Provide --model40-manifest or set SIDECHAIN_MODEL40_MANIFEST."
            )
        # Non-model40 width-38 model: leave unchanged.
        return x

    x_scaled = np.asarray(x, dtype=np.float32)
    if not x_scaled.flags.writeable:
        x_scaled = np.array(x_scaled, dtype=np.float32, copy=True)

    residue_col = int(cfg["residue_col"])
    bead_col = int(cfg["bead_col"])
    residue_offset = int(cfg["residue_offset"])
    num_res_types = int(cfg["num_res_types"])
    max_beads = int(cfg["max_beads"])

    if x_scaled.ndim != 2:
        raise ValueError(f"Expected 2D features for scaling, got shape {x_scaled.shape}")
    if residue_col < 0 or bead_col < 0 or residue_col >= x_scaled.shape[1] or bead_col >= x_scaled.shape[1]:
        raise ValueError(
            f"Scaling columns out of range for X shape {x_scaled.shape}: "
            f"residue_col={residue_col}, bead_col={bead_col}"
        )

    residue_scale = float(max(num_res_types - 1, 1))
    bead_scale = float(max(max_beads - 1, 1))

    x_scaled[:, residue_col] = np.clip(
        (x_scaled[:, residue_col] - float(residue_offset)) / residue_scale,
        0.0,
        1.0,
    )
    x_scaled[:, bead_col] = np.clip(
        x_scaled[:, bead_col] / bead_scale,
        0.0,
        1.0,
    )

    print(
        "Applied model_40 input scaling using manifest:",
        cfg["manifest_path"],
        f"(residue_col={residue_col}, bead_col={bead_col})",
    )
    return x_scaled


def normalize_mask(m):
    m = np.asarray(m, dtype=np.float32)
    if m.ndim != 2:
        return m

    # local_frames_AA padding sentinel convention.
    if np.any(np.isclose(m, -2.0)):
        return (m != -2.0).astype(np.float32)

    # Fallback for legacy encodings with arbitrary values.
    if np.any((m < 0.0) | (m > 1.0)):
        return (m > 0.0).astype(np.float32)

    return m


def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Missing model file: {args.model}")

    x_path, y_path, m_path = resolve_inputs(args)
    for p in (x_path,):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing required file: {p}")
    if not args.no_eval:
        for p in (y_path, m_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Missing required file: {p}")

    custom_objects = _load_optional_custom_objects()
    _install_dense_quantization_config_compat_patch()
    try:
        model = tf.keras.models.load_model(args.model, compile=False, custom_objects=custom_objects)
    except TypeError as e:
        msg = str(e)
        if "CoordMainBondAuxModel" in msg and "missing 3 required keyword-only arguments" in msg:
            raise TypeError(
                "Failed to load model_40 training checkpoint for inference.\n"
                "Use the model_40 portable inference artifact instead (e.g. *_bond_portable.keras), "
                "not ckpt_epoch_*_40_bond_*.keras."
            ) from e
        if "quantization_config" in msg:
            raise TypeError(
                "Failed to load the side-chain model because this Keras environment does not "
                "understand the saved Dense quantization_config field. The field is usually "
                "created by a newer Keras version. Use the same/newer TensorFlow/Keras environment "
                "used to export the portable model, or re-export the portable .keras model from "
                "the environment you will use for inference."
            ) from e
        raise

    x = load_arr(x_path, mmap_npy=True)
    expected_width = int(model.input_shape[-1])
    x = adapt_feature_width(x, expected_width)
    x = maybe_apply_model40_input_scaling(
        x=x,
        expected_width=expected_width,
        model_path=args.model,
        explicit_manifest=args.model40_manifest,
    )

    y = None
    m = None
    old_sample_num = None
    if not args.no_eval:
        y = load_arr(y_path, mmap_npy=True)
        m = load_arr(m_path, mmap_npy=True, require_arr_key=m_path.endswith(".npz"))
        old_sample_num = int(y.shape[0]) if y.ndim >= 1 else None

        if y.ndim == 2 and y.shape[1] % 15 == 0 and y.shape[0] != x.shape[0]:
            y = y.reshape(-1, 15).astype(np.float32)
        if m.ndim == 2 and m.shape[1] % 15 == 0 and m.shape[0] != x.shape[0]:
            m = m.reshape(-1, 15).astype(np.float32)
        m = normalize_mask(m)

        if y.shape[0] != x.shape[0] or m.shape[0] != x.shape[0]:
            raise ValueError(f"Sample mismatch: X={x.shape}, Y={y.shape}, M={m.shape}")

        # Use user-provided custom geometry loss/metrics for evaluation, not Keras defaults.
        geom = _resolve_geometry_eval_objects(custom_objects)
        loss_fn = geom["GeometryAwareMaskedMSE"]()
        coord_mse = geom["GeometryAwareCoordMSE"]()
        coord_rmse = geom["GeometryAwareCoordRMSE"]()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss=loss_fn,
            weighted_metrics=[coord_mse, coord_rmse],
        )

    print("Using model:", args.model)
    print("Using feature:", x_path, "shape=", x.shape)
    if not args.no_eval:
        print("Using target :", y_path, "shape=", y.shape)
        print("Using mask   :", m_path, "shape=", m.shape)
        print("Evaluation mode: custom geometry-aware loss/metrics")
        model.evaluate(x, y, sample_weight=m, batch_size=args.batch_size, verbose=1)

    y_pred = model.predict(x, batch_size=args.batch_size, verbose=2)
    print("Prediction shape:", y_pred.shape)

    out_file = pick_output_name(args)
    np.save(out_file, y_pred)
    print("Saved:", out_file)

    # Save frame-wise reshaped output for downstream denorm.
    if y_pred.ndim == 2:
        if old_sample_num is None and not args.no_eval and y is not None:
            old_sample_num = int(y.shape[0])
        if old_sample_num and y_pred.size % old_sample_num == 0:
            yhat_reshaped = y_pred.reshape(old_sample_num, -1).astype(np.float32)
            np.save("expert_Yhat_reshaped.npy", yhat_reshaped)
            print("Saved: expert_Yhat_reshaped.npy", yhat_reshaped.shape)


if __name__ == "__main__":
    main()
