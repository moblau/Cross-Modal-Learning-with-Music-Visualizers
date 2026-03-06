import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


AUDIO_NPZ = Path("mvt_audio_dataset_4fps.npz")
COLOR_NPZ = Path("mvt_colors_dataset_4fps.npz")
OUT_MODEL = Path("mvt_rf_model.joblib")

Y_TARGET = "weight motion"

N_SPLITS = 5
N_ESTIMATORS = 300
MAX_DEPTH = 12
MIN_SAMPLES_LEAF = 50
MAX_FEATURES = "sqrt"
RANDOM_STATE = 0


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

    Ysrc = c["Y"] if has_new else c["C"]

    a_idx = {n: i for i, n in enumerate(a_names)}
    c_idx = {n: i for i, n in enumerate(c_names)}
    common = sorted(set(a_idx) & set(c_idx))
    if not common:
        raise RuntimeError("No matching video names between audio and video datasets.")

    X_list, Y_list, groups = [], [], []

    for g, name in enumerate(common):
        i = a_idx[name]
        j = c_idx[name]

        n = int(min(a_len[i], c_len[j]))
        if n <= 0:
            continue

        Xi = X[i, :n, :]

        if has_new:
            Yi = Ysrc[j, :n, :]
        else:
            Yi = Ysrc[j, :n, 0, :]

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


def main():
    if not AUDIO_NPZ.exists() or not COLOR_NPZ.exists():
        raise SystemExit("Missing dataset files")

    X, Y_raw, groups, fmt = load_and_align(AUDIO_NPZ, COLOR_NPZ)

    if fmt == "new":
        Y = select_Y_new(Y_raw, Y_TARGET)
    else:
        Y = select_Y_legacy(Y_raw, Y_TARGET)

    print("Dataset:")
    print("  X:", X.shape)
    print("  Y_raw:", Y_raw.shape, f"(format={fmt})")
    print("  Y_selected:", Y.shape, f"(Y_TARGET='{Y_TARGET}')")
    print("  Videos:", len(set(groups)))
    print("  CV folds:", N_SPLITS)

    gkf = GroupKFold(n_splits=N_SPLITS)

    r2_overall_folds = []
    r2_each_folds = []

    for fold, (tr, te) in enumerate(gkf.split(X, Y, groups), start=1):
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        model.fit(X[tr], Y[tr])
        pred = model.predict(X[te])

        r2_overall = float(r2_score(Y[te], pred, multioutput="uniform_average"))
        r2_each = r2_score(Y[te], pred, multioutput="raw_values")

        r2_overall_folds.append(r2_overall)
        r2_each_folds.append(r2_each)

        print(
            f"Fold {fold}: R2 overall = {r2_overall:+.4f} | "
            f"per-dim = {np.array2string(r2_each, precision=4, separator=', ')}"
        )

    r2_overall_folds = np.array(r2_overall_folds, dtype=float)
    r2_each_folds = np.stack(r2_each_folds, axis=0).astype(float)

    print("\nGroupKFold CV results (by video):")
    print(f"  R2 overall: {r2_overall_folds.mean():+.4f} ± {r2_overall_folds.std():.4f}")
    print(f"  R2 per-dim mean: {np.array2string(r2_each_folds.mean(axis=0), precision=4, separator=', ')}")
    print(f"  R2 per-dim std : {np.array2string(r2_each_folds.std(axis=0),  precision=4, separator=', ')}")

    final_model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    final_model.fit(X, Y)

    importances = final_model.feature_importances_
    top = np.argsort(importances)[::-1][:10]

    print("\nTop 10 audio features (final model):")
    for idx in top:
        i = int(idx)
        print(f"  {feature_label(i):>20}: {importances[i]:.4f}")

    joblib.dump(
        {
            "model": final_model,
            "y_target": Y_TARGET,
            "color_format": fmt,
            "audio_feature_dim": int(X.shape[1]),
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "max_features": MAX_FEATURES,
            "random_state": RANDOM_STATE,
            "cv_r2_overall_folds": r2_overall_folds,
            "cv_r2_overall_mean": float(r2_overall_folds.mean()),
            "cv_r2_overall_std": float(r2_overall_folds.std()),
            "cv_r2_each_mean": r2_each_folds.mean(axis=0),
            "cv_r2_each_std": r2_each_folds.std(axis=0),
        },
        OUT_MODEL
    )

    print(f"\n✅ Saved final model: {OUT_MODEL.resolve()}")


if __name__ == "__main__":
    main()