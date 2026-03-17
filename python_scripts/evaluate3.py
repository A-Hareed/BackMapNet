import tensorflow as tf
import numpy as np
import sys
import os
import re
import glob
import gc
import zipfile
import tempfile
import shutil
import json
import inspect
from tensorflow.keras.optimizers import Adam
from metric_function_BB import make_custom_objects, resolve_prior_path
from final_model_activation_test import build_1d_conv_autoencoder_multi_input


raw_args = [a for a in sys.argv[1:] if a not in ("--cg-only", "--load-full-model")]
cg_only = "--cg-only" in sys.argv[1:]
load_full_model = "--load-full-model" in sys.argv[1:]

batch_size = 5000
json_weights_path = None
filtered_args = []
i = 0
while i < len(raw_args):
    if raw_args[i] == "--batch-size":
        if i + 1 >= len(raw_args):
            raise SystemExit("Missing value after --batch-size")
        batch_size = int(raw_args[i + 1])
        i += 2
    elif raw_args[i] == "--json-weights":
        if i + 1 >= len(raw_args):
            raise SystemExit("Missing value after --json-weights")
        json_weights_path = raw_args[i + 1]
        i += 2
    else:
        filtered_args.append(raw_args[i])
        i += 1

if len(filtered_args) < 3:
    raise SystemExit(
        "Usage: python evaluate3.py <pdb_name> <chain_num|ALL> <model_path> "
        "[--cg-only] [--batch-size N] [--load-full-model] [--json-weights <path>]"
    )

pdb = filtered_args[0]
chain_selector = filtered_args[1]
model_path = filtered_args[2]
if batch_size <= 0:
    raise SystemExit("--batch-size must be a positive integer")
print("chain selector:", chain_selector)
print("batch size:", batch_size)

#boxsize = np.array([22.50000,  22.50000,  22.50000]) *10


def resolve_model_path(path_text):
    """
    Resolve model path from:
    1) exact path as provided
    2) relative to current working directory
    3) relative to script directory
    """
    p = os.path.expanduser(path_text)
    if os.path.exists(p):
        return os.path.abspath(p)

    p_cwd = os.path.abspath(path_text)
    if os.path.exists(p_cwd):
        return p_cwd

    script_dir = os.path.dirname(os.path.abspath(__file__))
    p_script = os.path.join(script_dir, path_text)
    if os.path.exists(p_script):
        return os.path.abspath(p_script)

    raise FileNotFoundError(
        f"Model file not found: {path_text}\n"
        f"cwd={os.getcwd()}\n"
        f"script_dir={script_dir}"
    )


def maybe_make_h5_alias(model_file_path):
    """
    Keras 3 expects '.keras' paths to be zip-format.
    If file is actually legacy HDF5 with a '.keras' name,
    create a temporary '.h5' alias and load via that path.
    """
    needs_cleanup = None
    path_for_load = model_file_path

    if model_file_path.endswith(".keras") and not zipfile.is_zipfile(model_file_path):
        is_h5 = False
        try:
            import h5py
            is_h5 = h5py.is_hdf5(model_file_path)
        except Exception:
            is_h5 = False

        if is_h5:
            tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
            tmp.close()
            shutil.copy2(model_file_path, tmp.name)
            path_for_load = tmp.name
            needs_cleanup = tmp.name

    return path_for_load, needs_cleanup


def _unique_keep_order(items):
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def resolve_json_weights_path(json_arch_path, explicit_weights_path=None):
    if explicit_weights_path:
        return resolve_model_path(explicit_weights_path)

    json_abs = os.path.abspath(json_arch_path)
    json_dir = os.path.dirname(json_abs)
    json_base = os.path.splitext(os.path.basename(json_abs))[0]

    stems = [json_base]
    if json_base.endswith(".arch"):
        stems.append(json_base[:-5])

    candidates = []
    for stem in stems:
        candidates.extend(
            [
                os.path.join(json_dir, f"{stem}.weights.h5"),
                os.path.join(json_dir, f"{stem}.h5"),
                os.path.join(json_dir, f"{stem}_multi_input.weights.h5"),
                os.path.join(json_dir, f"{stem}_multi_input.h5"),
            ]
        )

    for candidate in _unique_keep_order(candidates):
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(
        "JSON architecture provided but no matching weights file was found.\n"
        f"Architecture: {json_arch_path}\n"
        "Provide --json-weights <path> or place one of these files next to the JSON:\n"
        + "\n".join(f"  - {x}" for x in _unique_keep_order(candidates))
    )


def patch_keras2_json_for_keras3(obj):
    if isinstance(obj, dict):
        if "class_name" in obj and "config" in obj and isinstance(obj["config"], dict):
            cls_name = obj["class_name"]
            cfg = obj["config"]

            if cls_name == "BatchNormalization":
                axis = cfg.get("axis")
                if isinstance(axis, list) and len(axis) == 1:
                    cfg["axis"] = axis[0]

            if cls_name == "LeakyReLU":
                if "alpha" in cfg and "negative_slope" not in cfg:
                    cfg["negative_slope"] = cfg.pop("alpha")

            if cls_name in ("Conv1DTranspose", "Conv2DTranspose", "Conv3DTranspose"):
                cfg.pop("groups", None)

            if cls_name == "InputLayer":
                if "batch_input_shape" in cfg and "batch_shape" not in cfg:
                    cfg["batch_shape"] = cfg.pop("batch_input_shape")

            layer_cls = getattr(tf.keras.layers, cls_name, None)
            if layer_cls is not None:
                try:
                    sig = inspect.signature(layer_cls.__init__)
                    has_var_kwargs = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in sig.parameters.values()
                    )
                    if not has_var_kwargs:
                        allowed = {k for k in sig.parameters if k != "self"}
                        obj["config"] = {k: v for k, v in cfg.items() if k in allowed}
                except (TypeError, ValueError):
                    pass

        for value in obj.values():
            patch_keras2_json_for_keras3(value)
    elif isinstance(obj, list):
        for value in obj:
            patch_keras2_json_for_keras3(value)


def load_model_from_json_arch(json_arch_path, custom_objects_dict):
    with open(json_arch_path, "r") as fh:
        arch = json.load(fh)
    patch_keras2_json_for_keras3(arch)

    json_custom_objects = dict(custom_objects_dict)
    try:
        from keras.src.models.functional import Functional

        json_custom_objects["Functional"] = Functional
    except Exception:
        pass

    return tf.keras.models.model_from_json(
        json.dumps(arch),
        custom_objects=json_custom_objects,
    )


print('current model being loaded')

prior_path = resolve_prior_path()
custom_objects = make_custom_objects(prior_path=prior_path, percentile=14)
print("Rama prior:", prior_path)

combined_coord_and_torsion_loss = custom_objects["combined_coord_and_torsion_loss"]
CoordRMSE = custom_objects["CoordRMSE"]
rama_penalty = custom_objects["rama_penalty"]
normalized_coord_mse = custom_objects["normalized_coord_mse"]
frac_metric = custom_objects["frac_metric"]


resolved_model_path = resolve_model_path(model_path)
if resolved_model_path.lower().endswith(".json"):
    resolved_json_weights_path = resolve_json_weights_path(
        resolved_model_path, explicit_weights_path=json_weights_path
    )
    try:
        refinement_model_multi_input = load_model_from_json_arch(
            resolved_model_path,
            custom_objects,
        )
        _ = refinement_model_multi_input(
            [
                tf.zeros((1, 96), dtype=tf.float32),
                tf.zeros((1, 1, 3), dtype=tf.float32),
            ]
        )
        refinement_model_multi_input.load_weights(resolved_json_weights_path)
        print(f"Loaded JSON architecture: {resolved_model_path}")
        print(f"Loaded model weights: {resolved_json_weights_path}")
    except Exception as json_load_error:
        raise ValueError(
            f"Failed to load model from JSON architecture: {resolved_model_path}\n"
            f"Weights path used: {resolved_json_weights_path}\n"
            "If your weights file does not follow standard naming, pass --json-weights <path>.\n"
            f"Original error: {json_load_error}"
        ) from json_load_error
elif load_full_model:
    load_model_path, temp_h5_alias = maybe_make_h5_alias(resolved_model_path)
    try:
        refinement_model_multi_input = tf.keras.models.load_model(
            load_model_path,
            custom_objects=custom_objects,
            compile=False,
        )
        print(f"Loaded full model: {resolved_model_path}")
    finally:
        if temp_h5_alias is not None and os.path.exists(temp_h5_alias):
            os.remove(temp_h5_alias)
else:
    refinement_model_multi_input = build_1d_conv_autoencoder_multi_input(96)
    _ = refinement_model_multi_input(
        [
            tf.zeros((1, 96), dtype=tf.float32),
            tf.zeros((1, 1, 3), dtype=tf.float32),
        ]
    )
    try:
        refinement_model_multi_input.load_weights(resolved_model_path)
        print(f"Loaded model weights: {resolved_model_path}")
    except Exception as weight_error:
        raise ValueError(
            f"Failed to load weights from: {resolved_model_path}\n"
            "Default mode expects model weights (.h5/.weights.h5) for the multi-input architecture.\n"
            "If you intentionally want to load a full saved model, rerun with: --load-full-model\n"
            f"Original error: {weight_error}"
        ) from weight_error




optimizer = Adam(learning_rate=1e-4)

#model = tf.keras.models.load_model(sys.argv[3])

#print(refinement_model_multi_input.summary())

chain_re = re.compile(rf"^train_feat_B(\d+)_{re.escape(pdb)}_chain(\d+)\.npy$")


def discover_chains():
    if chain_selector.upper() != "ALL":
        return [int(chain_selector)]
    chains = set()
    for fp in glob.glob(f"train_feat_B*_{pdb}_chain*.npy"):
        name = os.path.basename(fp)
        m = chain_re.match(name)
        if m:
            chains.add(int(m.group(2)))
    return sorted(chains)


chains_to_run = discover_chains()
if not chains_to_run:
    raise FileNotFoundError(
        f"No chain inputs found for pdb={pdb}. Expected files like "
        f"train_feat_B<idx>_{pdb}_chain<chain>.npy"
    )

# Compile once, only if at least one chain has labels and evaluation is requested.
if not cg_only:
    any_labels = any(
        glob.glob(f"train_LAB_B*_{pdb}_chain{chain}.npy")
        for chain in chains_to_run
    )
    if any_labels:
        refinement_model_multi_input.compile(
            optimizer=optimizer,
            loss=combined_coord_and_torsion_loss,
            metrics=[CoordRMSE(name='coord_rmse'),rama_penalty,normalized_coord_mse,frac_metric],
        )

for chain_num in chains_to_run:
    print("processing chain:", chain_num)
    pattern = f"train_feat_B*_{pdb}_chain{chain_num}.npy"
    regex = re.compile(rf"^train_feat_B(\d+)_{re.escape(pdb)}_chain{chain_num}\.npy$")
    frames_with_labels = []
    frame_indices = []
    for fp in glob.glob(pattern):
        name = os.path.basename(fp)
        m = regex.match(name)
        if m:
            idx = int(m.group(1))
            range_fp = f"custom_range_B{idx}_{pdb}_chain{chain_num}.npy"
            lab_fp = f"train_LAB_B{idx}_{pdb}_chain{chain_num}.npy"
            if os.path.exists(range_fp):
                frame_indices.append(idx)
                if os.path.exists(lab_fp):
                    frames_with_labels.append(idx)

    frame_indices = sorted(set(frame_indices))
    frames_with_labels = sorted(set(frames_with_labels))
    if not frame_indices:
        raise FileNotFoundError(
            f"No frame inputs found for pdb={pdb}, chain={chain_num}. "
            f"Expected files like train_feat_B<idx>_{pdb}_chain{chain_num}.npy"
        )

    do_evaluate = (not cg_only) and len(frames_with_labels) > 0
    for i in frame_indices:
        feat_arr = np.load(
            f'train_feat_B{i}_{pdb}_chain{chain_num}.npy',
            mmap_mode="r",
        )
        range_arr = np.load(
            f'custom_range_B{i}_{pdb}_chain{chain_num}.npy',
            mmap_mode="r",
        )
        lab_arr = None
        if do_evaluate and i in frames_with_labels:
            lab_arr = np.load(
                f'train_LAB_B{i}_{pdb}_chain{chain_num}.npy',
                mmap_mode="r",
            )
            refinement_model_multi_input.evaluate(
                [feat_arr, range_arr],
                lab_arr,
                verbose=1,
                batch_size=batch_size,
            )
        yhat = refinement_model_multi_input.predict(
            [feat_arr, range_arr],
            verbose=2,
            batch_size=batch_size,
        )
        yhat = yhat[:, :384].astype(np.float64, copy=False)
        np.save(f'RAMAPROIR_yhat_frame_{i}_chain_{chain_num}.npy', yhat)
        del yhat
        del feat_arr
        del range_arr
        if lab_arr is not None:
            del lab_arr
        gc.collect()

exit()
#
#    else:
#        feat_arr = np.concatenate((feat_arr,np.load(feat_lst[i])),axis=0)
#        lab_arr = np.concatenate((lab_arr, np.load(lab_lst[i])),axis =0)
#    print(feat_arr.shape)

#"""
#feat_arr = np.load(feat_lst[0])
#lab_arr = np.load(lab_lst[0])



#model = tf.keras.models.load_model('model5_check_epoch_03.keras')
#model = tf.keras.models.load_model(sys.argv[3])

# Assume validation data is loaded in variables X_val and y_val
# X_val: features, y_val: true labels




# Make predictions on the validation data
history = model.evaluate(feat_arr,lab_arr)

yhat = model.predict(feat_arr)

np.save(f'yhat_whole_{frame_num}.npy',yhat)

print(history)


exit()
