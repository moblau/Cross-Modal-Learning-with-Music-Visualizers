import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error


AUDIO_NPZ = Path("mvt_audio_dataset_4fps.npz")
COLOR_NPZ = Path("mvt_colors_dataset_4fps.npz")
OUT_MODEL = Path("mvt_boost_model.joblib")

Y_TARGET = "weight motion"

N_SPLITS = 5
RANDOM_STATE = 0

MAX_ITER = 600
LEARNING_RATE = 0.05
MAX_DEPTH = 3
MAX_LEAF_NODES = 31
MIN_SAMPLES_LEAF = 50
L2_REGULARIZATION = 1e-3
EARLY_STOPPING = True
VALIDATION_FRACTION = 0.1
N_ITER_NO_CHANGE = 25


def _as_str_list(arr):
    return [str(x) for x in arr.tolist()]


def select_Y_new(Y: np.ndarray, target: str) -> np.ndarray:
    feature_map = {
        "weight": [0],
        "hue": [1, 2],
        "bright": [3],
        "brightness": [3],
        "lab": [4, 5, 6],
        "l": [4],
        "a": [5],
        "b": [6],
        "motion": [7],
    }

    tokens = target.lower().split()
    if not tokens:
        raise ValueError("Y_TARGET is empty.")

    cols = []
    for tok in tokens:
        if tok not in feature_map:
            raise ValueError(f"Unknown token '{tok}'. Valid tokens (new): {sorted(feature_map)}")
        cols.extend(feature_map[tok])

    cols = list(dict.fromkeys(cols))
    return Y[:, cols]


def select_Y_legacy(C_flat: np.ndarray, target: str) -> np.ndarray:
    feature_map = {
        "weight": [0],
        "hue": [1, 2],
        "s": [3],
        "v": [4],
        "sv": [3, 4],
    }

    tokens = target.lower().split()
    if not tokens:
        raise ValueError("Y_TARGET is empty.")

    cols = []
    for tok in tokens:
        if tok not in feature_map:
            raise ValueError(f"Unknown token '{tok}'. Valid tokens (legacy): {sorted(feature_map)}")
        cols.extend(feature_map[tok])

    cols = list(dict.fromkeys(cols))
    return C_flat[:, cols]


def feature_label(i: int) -> str:
    if 0 <= i <= 63:
        return f"mel_{i}"
    mapping = {
        64: "rms_mean",
        65: "centroid_mean",
        66: "rolloff85_mean",
        67: "bandwidth_mean",
        68: "onset_mean",
        69: "mel_low_mean",
        70: "mel_mid_mean",
        71: "mel_high_mean",
        72: "delta_rms",
        73: "delta_centroid",
        74: "delta_low",
        75: "delta_mid",
        76: "delta_high",
    }
    return mapping.get(i, f"feature_{i}")


def load_and_align(audio_path: Path, color_path: Path):
    a = np.load(audio_path, allow_pickle=True)
    c = np.load(color_path, allow_pickle=True)

    X = a["X"]
    a_names = _as_str_list(a["names"])
    a_len = a["lengths"]

    c_names = _as_str_list(c["names"])
    c_len = c["lengths"]

    has_new = "Y" in c.files
    has_legacy = "C" in c.files
    if not (has_new or has_legacy):
        raise KeyError(f"Color NPZ has no 'Y' or 'C'. Keys: {c.files}")

    if has_new:
        Ysrc = c["Y"]
    else:
        Csrc = c["C"]

    a_idx = {n: i for i, n in enumerate(a_names)}
    c_idx = {n: i for i, n in enumerate(c_names)}
    common = sorted(set(a_idx) & set(c_idx))
    if not common:
        raise RuntimeError("No matching video names between audio and color datasets.")

    X_list, Y_list, groups = [], [], []

    for g, name in enumerate(common):
        i = a_idx[name]
        j = c_idx[name]

        n = int(min(a_len[i], c_len[j]))
        if n <= 0:
            continue

        Xi = X[i, :n, :]
        Yi = Ysrc[j, :n, :] if has_new else Csrc[j, :n, 0, :]

        good = np.isfinite(Xi).all(axis=1) & np.isfinite(Yi).all(axis=1)
        Xi = Xi[good]
        Yi = Yi[good]

        if len(Xi) == 0:
            continue

        X_list.append(Xi)
        Y_list.append(Yi)
        groups.append(np.full(len(Xi), g, dtype=int))

    if not X_list:
        raise RuntimeError("After filtering, no valid aligned rows remained.")

    X_all = np.vstack(X_list)
    Y_all = np.vstack(Y_list)
    groups_all = np.concatenate(groups)

    fmt = "new" if has_new else "legacy"
    return X_all, Y_all, groups_all, fmt


def make_booster():
    base = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=LEARNING_RATE,
        max_iter=MAX_ITER,
        max_depth=MAX_DEPTH,
        max_leaf_nodes=MAX_LEAF_NODES,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        l2_regularization=L2_REGULARIZATION,
        early_stopping=EARLY_STOPPING,
        validation_fraction=VALIDATION_FRACTION,
        n_iter_no_change=N_ITER_NO_CHANGE,
        random_state=RANDOM_STATE,
    )
    return MultiOutputRegressor(base, n_jobs=-1)


def main():
    if not AUDIO_NPZ.exists() or not COLOR_NPZ.exists():
        raise SystemExit("Missing dataset files")

    X, Y_raw, groups, fmt = load_and_align(AUDIO_NPZ, COLOR_NPZ)

    Y = select_Y_new(Y_raw, Y_TARGET) if fmt == "new" else select_Y_legacy(Y_raw, Y_TARGET)

    print("Dataset:")
    print("  X:", X.shape)
    print("  Y_raw:", Y_raw.shape, f"(format={fmt})")
    print("  Y_selected:", Y.shape, f"(Y_TARGET='{Y_TARGET}')")
    print("  Videos:", len(set(groups)))
    print("  CV folds:", N_SPLITS)

    gkf = GroupKFold(n_splits=N_SPLITS)

    r2_overall_folds = []
    r2_each_folds = []
    rmse_folds = []

    for fold, (tr, te) in enumerate(gkf.split(X, Y, groups), start=1):
        model = make_booster()
        model.fit(X[tr], Y[tr])
        pred = model.predict(X[te])

        r2_overall = float(r2_score(Y[te], pred, multioutput="uniform_average"))
        r2_each = r2_score(Y[te], pred, multioutput="raw_values")
        rmse = float(np.sqrt(mean_squared_error(Y[te], pred)))

        r2_overall_folds.append(r2_overall)
        r2_each_folds.append(r2_each)
        rmse_folds.append(rmse)

        print(
            f"Fold {fold}: R2 overall = {r2_overall:+.4f} | "
            f"per-dim = {np.array2string(r2_each, precision=4, separator=', ')} | "
            f"RMSE = {rmse:.4f}"
        )

    r2_overall_folds = np.array(r2_overall_folds, dtype=float)
    r2_each_folds = np.stack(r2_each_folds, axis=0).astype(float)
    rmse_folds = np.array(rmse_folds, dtype=float)

    print("\nGroupKFold CV results (by video):")
    print(f"  R2 overall: {r2_overall_folds.mean():+.4f} ± {r2_overall_folds.std():.4f}")
    print(f"  R2 per-dim mean: {np.array2string(r2_each_folds.mean(axis=0), precision=4, separator=', ')}")
    print(f"  R2 per-dim std : {np.array2string(r2_each_folds.std(axis=0),  precision=4, separator=', ')}")
    print(f"  RMSE mean±std : {rmse_folds.mean():.4f} ± {rmse_folds.std():.4f}")

    final_model = make_booster()
    final_model.fit(X, Y)

    joblib.dump(
        {
            "model": final_model,
            "y_target": Y_TARGET,
            "color_format": fmt,
            "audio_feature_dim": int(X.shape[1]),
            "booster": "HistGradientBoostingRegressor",
            "params": {
                "max_iter": MAX_ITER,
                "learning_rate": LEARNING_RATE,
                "max_depth": MAX_DEPTH,
                "max_leaf_nodes": MAX_LEAF_NODES,
                "min_samples_leaf": MIN_SAMPLES_LEAF,
                "l2_regularization": L2_REGULARIZATION,
                "early_stopping": EARLY_STOPPING,
                "validation_fraction": VALIDATION_FRACTION,
                "n_iter_no_change": N_ITER_NO_CHANGE,
                "random_state": RANDOM_STATE,
            },
            "cv_r2_overall_folds": r2_overall_folds,
            "cv_r2_overall_mean": float(r2_overall_folds.mean()),
            "cv_r2_overall_std": float(r2_overall_folds.std()),
            "cv_r2_each_mean": r2_each_folds.mean(axis=0),
            "cv_r2_each_std": r2_each_folds.std(axis=0),
            "cv_rmse_folds": rmse_folds,
            "cv_rmse_mean": float(rmse_folds.mean()),
            "cv_rmse_std": float(rmse_folds.std()),
        },
        OUT_MODEL,
    )

    print(f"\n✅ Saved final model: {OUT_MODEL.resolve()}")


if __name__ == "__main__":
    main()