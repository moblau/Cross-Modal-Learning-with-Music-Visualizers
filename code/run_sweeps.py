import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor


DATA_DIR = Path("datasets_multi_fps")
OUT_DIR = Path("fps_sweep_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FPS_LIST = [8]

AUDIO_FMT = "mvt_audio_dataset_fps{fps}.npz"
VIDEO_FMT = "mvt_colors_dataset_fps{fps}.npz"

Y_TARGET = "weight motion"
N_SPLITS = 5

RUN_RIDGE = False
RUN_RF = False
RUN_BOOST = True

RIDGE_ALPHAS = [10**p for p in range(-2, 7)]

RF_PARAMS = dict(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=10,
    max_features="sqrt",
    n_jobs=-1,
    random_state=0,
)

RF_TUNE = True
RF_TUNE_FPS = "best"
RF_TUNE_N_ESTIMATORS = 300

RF_LEAF_LIST = [1, 2, 5, 10, 20, 50, 100]
RF_MAXFEAT_LIST = ["sqrt", 0.3, 0.5, 0.7, 1.0]
RF_DEPTH_LIST = [None, 10, 20, 30]
RF_TUNE_DEPTH = False

BOOST_PARAMS = dict(
    max_depth=6,
    learning_rate=0.05,
    max_iter=400,
    random_state=0,
)

BOOST_TUNE = True
BOOST_TUNE_FPS = "best"
BOOST_TUNE_MAX_ITER = 200
BOOST_TUNE_MAX_ITER_FINAL = True

BOOST_LR_LIST = [0.01]
BOOST_DEPTH_LIST = [4]
BOOST_LEAF_LIST = [31]
BOOST_L2_LIST = [1e-1]
BOOST_SUBSAMPLE_LIST = [0.8]
BOOST_MAXITER_LIST = [400]


def _as_str_list(arr):
    return [str(x) for x in arr.tolist()]


def select_y_cols(Y_raw: np.ndarray, y_target: str) -> np.ndarray:
    tokens = y_target.strip().lower().split()
    valid = {"weight": 0, "motion": 1}
    cols = [valid[t] for t in tokens]
    Y = Y_raw[:, cols]
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return Y.astype(np.float32)


def load_and_align(audio_npz: Path, video_npz: Path, y_target: str):
    a = np.load(audio_npz, allow_pickle=True)
    v = np.load(video_npz, allow_pickle=True)

    X = a["X"]
    a_names = _as_str_list(a["names"])
    a_len = a["lengths"].astype(int)

    Yv = v["Y"]
    v_names = _as_str_list(v["names"])
    v_len = v["lengths"].astype(int)

    a_idx = {n: i for i, n in enumerate(a_names)}
    v_idx = {n: i for i, n in enumerate(v_names)}
    common = sorted(set(a_idx) & set(v_idx))

    X_list, Y_list, groups_list = [], [], []

    for g, name in enumerate(common):
        i = a_idx[name]
        j = v_idx[name]
        n = int(min(a_len[i], v_len[j]))
        if n <= 1:
            continue

        Xi = X[i, :n, :]
        Yi_raw = Yv[j, :n, :]

        good = np.isfinite(Xi).all(axis=1) & np.isfinite(Yi_raw).all(axis=1)
        Xi = Xi[good]
        Yi_raw = Yi_raw[good]
        if len(Xi) == 0:
            continue

        Yi = select_y_cols(Yi_raw, y_target)

        X_list.append(Xi)
        Y_list.append(Yi)
        groups_list.append(np.full(len(Xi), g, dtype=int))

    X_all = np.vstack(X_list).astype(np.float32)
    Y_all = np.vstack(Y_list).astype(np.float32)
    groups = np.concatenate(groups_list).astype(int)

    return X_all, Y_all, groups, common


def cv_r2_rmse(model, X, Y, groups, n_splits: int):
    gkf = GroupKFold(n_splits=n_splits)

    r2_overall = []
    r2_per_dim = []
    rmse_list = []

    for tr, te in gkf.split(X, Y, groups):
        model.fit(X[tr], Y[tr])
        pred = model.predict(X[te])

        r2_overall.append(float(r2_score(Y[te], pred, multioutput="uniform_average")))
        r2_per_dim.append(np.array(r2_score(Y[te], pred, multioutput="raw_values"), dtype=float))
        rmse_list.append(float(np.sqrt(mean_squared_error(Y[te], pred))))

    r2_overall = np.array(r2_overall)
    r2_per_dim = np.vstack(r2_per_dim)
    rmse_list = np.array(rmse_list)

    return {
        "r2_mean": float(r2_overall.mean()),
        "r2_std": float(r2_overall.std()),
        "r2_per_dim_mean": r2_per_dim.mean(axis=0).tolist(),
        "r2_per_dim_std": r2_per_dim.std(axis=0).tolist(),
        "rmse_mean": float(rmse_list.mean()),
        "rmse_std": float(rmse_list.std()),
    }


def rf_param_sweep(X, Y, groups, base_params: dict, param_name: str, values: list, n_estimators_override: int):
    curve = []
    for v in values:
        params = dict(base_params)
        params[param_name] = v
        params["n_estimators"] = n_estimators_override
        stats = cv_r2_rmse(RandomForestRegressor(**params), X, Y, groups, N_SPLITS)
        curve.append({param_name: v, **stats})
    return curve


def boost_param_sweep(X, Y, groups, base_params: dict, param_name: str, values: list, max_iter_override: int):
    curve = []
    for v in values:
        params = dict(base_params)
        params[param_name] = v
        params["max_iter"] = max_iter_override
        model = MultiOutputRegressor(HistGradientBoostingRegressor(**params))
        stats = cv_r2_rmse(model, X, Y, groups, N_SPLITS)
        curve.append({param_name: v, **stats})
    return curve


def plot_1d_curve(values, r2s, xlabel, title, out_path: Path, xscale=None):
    plt.figure()
    plt.plot(values, r2s, marker="o")
    if xscale:
        plt.xscale(xscale)
    plt.xlabel(xlabel)
    plt.ylabel("CV R2 (mean)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def plot_1d_curve_cat(x_labels, r2s, xlabel, title, out_path: Path):
    plt.figure()
    x_pos = np.arange(len(x_labels))
    plt.plot(x_pos, r2s, marker="o")
    plt.xticks(x_pos, [str(x) for x in x_labels])
    plt.xlabel(xlabel)
    plt.ylabel("CV R2 (mean)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def main():
    results = {}

    for fps in FPS_LIST:
        audio_npz = DATA_DIR / AUDIO_FMT.format(fps=fps)
        video_npz = DATA_DIR / VIDEO_FMT.format(fps=fps)

        X, Y, groups, _ = load_and_align(audio_npz, video_npz, Y_TARGET)
        results[fps] = {"shape": {"X": list(X.shape), "Y": list(Y.shape)}, "models": {}}

        if RUN_RIDGE:
            ridge_curve = []
            best_alpha = None
            best_r2 = -1e9
            for alpha in RIDGE_ALPHAS:
                ridge_model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha, solver="svd")),
                ])
                stats = cv_r2_rmse(ridge_model, X, Y, groups, N_SPLITS)
                ridge_curve.append({"alpha": alpha, **stats})
                if stats["r2_mean"] > best_r2:
                    best_r2 = stats["r2_mean"]
                    best_alpha = alpha

            results[fps]["models"]["ridge"] = {
                "best_alpha": best_alpha,
                "curve": ridge_curve,
                "best_r2_mean": best_r2,
            }

        if RUN_RF:
            rf_stats = cv_r2_rmse(RandomForestRegressor(**RF_PARAMS), X, Y, groups, N_SPLITS)
            results[fps]["models"]["rf"] = {"params": RF_PARAMS, **rf_stats}

        if RUN_BOOST:
            base = HistGradientBoostingRegressor(**BOOST_PARAMS)
            boost_stats = cv_r2_rmse(MultiOutputRegressor(base), X, Y, groups, N_SPLITS)
            results[fps]["models"]["boost"] = {"params": BOOST_PARAMS, **boost_stats}

    if RUN_RF and RF_TUNE:
        if RF_TUNE_FPS == "best":
            tune_fps = max(
                [f for f in FPS_LIST if "rf" in results[f]["models"]],
                key=lambda f: results[f]["models"]["rf"]["r2_mean"],
            )
        else:
            tune_fps = int(RF_TUNE_FPS)

        audio_npz = DATA_DIR / AUDIO_FMT.format(fps=tune_fps)
        video_npz = DATA_DIR / VIDEO_FMT.format(fps=tune_fps)
        X, Y, groups, _ = load_and_align(audio_npz, video_npz, Y_TARGET)

        results[tune_fps]["models"].setdefault("rf_tuning", {})
        base_rf = dict(RF_PARAMS)

        leaf_curve = rf_param_sweep(X, Y, groups, base_rf, "min_samples_leaf", RF_LEAF_LIST, RF_TUNE_N_ESTIMATORS)
        results[tune_fps]["models"]["rf_tuning"]["min_samples_leaf"] = leaf_curve
        best_leaf = max(leaf_curve, key=lambda d: d["r2_mean"])["min_samples_leaf"]

        out_leaf = OUT_DIR / f"rf_tune_leaf_fps{tune_fps}.png"
        plot_1d_curve(
            [d["min_samples_leaf"] for d in leaf_curve],
            [d["r2_mean"] for d in leaf_curve],
            "min_samples_leaf",
            f"RF tuning @ FPS={tune_fps}",
            out_leaf,
            xscale="log" if max(RF_LEAF_LIST) / max(1, min(RF_LEAF_LIST)) >= 50 else None,
        )

        base_rf2 = dict(base_rf)
        base_rf2["min_samples_leaf"] = best_leaf
        maxfeat_curve = rf_param_sweep(X, Y, groups, base_rf2, "max_features", RF_MAXFEAT_LIST, RF_TUNE_N_ESTIMATORS)
        results[tune_fps]["models"]["rf_tuning"]["max_features"] = maxfeat_curve
        best_maxfeat = max(maxfeat_curve, key=lambda d: d["r2_mean"])["max_features"]

        out_maxfeat = OUT_DIR / f"rf_tune_max_features_fps{tune_fps}.png"
        plot_1d_curve_cat(
            [d["max_features"] for d in maxfeat_curve],
            [d["r2_mean"] for d in maxfeat_curve],
            "max_features",
            f"RF tuning @ FPS={tune_fps}",
            out_maxfeat,
        )

        if RF_TUNE_DEPTH:
            base_rf3 = dict(base_rf2)
            base_rf3["max_features"] = best_maxfeat
            depth_curve = rf_param_sweep(X, Y, groups, base_rf3, "max_depth", RF_DEPTH_LIST, RF_TUNE_N_ESTIMATORS)
            results[tune_fps]["models"]["rf_tuning"]["max_depth"] = depth_curve

            out_depth = OUT_DIR / f"rf_tune_max_depth_fps{tune_fps}.png"
            plot_1d_curve_cat(
                ["None" if d["max_depth"] is None else str(d["max_depth"]) for d in depth_curve],
                [d["r2_mean"] for d in depth_curve],
                "max_depth",
                f"RF tuning @ FPS={tune_fps}",
                out_depth,
            )

    if RUN_BOOST and BOOST_TUNE:
        if BOOST_TUNE_FPS == "best":
            tune_fps = max(
                [f for f in FPS_LIST if "boost" in results[f]["models"]],
                key=lambda f: results[f]["models"]["boost"]["r2_mean"],
            )
        else:
            tune_fps = int(BOOST_TUNE_FPS)

        audio_npz = DATA_DIR / AUDIO_FMT.format(fps=tune_fps)
        video_npz = DATA_DIR / VIDEO_FMT.format(fps=tune_fps)
        X, Y, groups, _ = load_and_align(audio_npz, video_npz, Y_TARGET)

        results[tune_fps]["models"].setdefault("boost_tuning", {})
        base_boost = dict(BOOST_PARAMS)

        lr_curve = boost_param_sweep(X, Y, groups, base_boost, "learning_rate", BOOST_LR_LIST, BOOST_TUNE_MAX_ITER)
        results[tune_fps]["models"]["boost_tuning"]["learning_rate"] = lr_curve
        best_lr = max(lr_curve, key=lambda d: d["r2_mean"])["learning_rate"]
        plot_1d_curve(
            BOOST_LR_LIST,
            [d["r2_mean"] for d in lr_curve],
            "learning_rate",
            f"BOOST tuning @ FPS={tune_fps}",
            OUT_DIR / f"boost_tune_learning_rate_fps{tune_fps}.png",
            xscale="log",
        )

        base_b2 = dict(base_boost)
        base_b2["learning_rate"] = best_lr
        depth_curve = boost_param_sweep(X, Y, groups, base_b2, "max_depth", BOOST_DEPTH_LIST, BOOST_TUNE_MAX_ITER)
        results[tune_fps]["models"]["boost_tuning"]["max_depth"] = depth_curve
        best_depth = max(depth_curve, key=lambda d: d["r2_mean"])["max_depth"]
        plot_1d_curve(
            BOOST_DEPTH_LIST,
            [d["r2_mean"] for d in depth_curve],
            "max_depth",
            f"BOOST tuning @ FPS={tune_fps}",
            OUT_DIR / f"boost_tune_max_depth_fps{tune_fps}.png",
        )

        base_b3 = dict(base_b2)
        base_b3["max_depth"] = best_depth
        leaf_curve = boost_param_sweep(X, Y, groups, base_b3, "max_leaf_nodes", BOOST_LEAF_LIST, BOOST_TUNE_MAX_ITER)
        results[tune_fps]["models"]["boost_tuning"]["max_leaf_nodes"] = leaf_curve
        best_leaf = max(leaf_curve, key=lambda d: d["r2_mean"])["max_leaf_nodes"]
        plot_1d_curve(
            BOOST_LEAF_LIST,
            [d["r2_mean"] for d in leaf_curve],
            "max_leaf_nodes",
            f"BOOST tuning @ FPS={tune_fps}",
            OUT_DIR / f"boost_tune_max_leaf_nodes_fps{tune_fps}.png",
            xscale="log" if max(BOOST_LEAF_LIST) / min(BOOST_LEAF_LIST) >= 50 else None,
        )

        base_b4 = dict(base_b3)
        base_b4["max_leaf_nodes"] = best_leaf
        l2_curve = boost_param_sweep(X, Y, groups, base_b4, "l2_regularization", BOOST_L2_LIST, BOOST_TUNE_MAX_ITER)
        results[tune_fps]["models"]["boost_tuning"]["l2_regularization"] = l2_curve
        best_l2 = max(l2_curve, key=lambda d: d["r2_mean"])["l2_regularization"]
        plot_1d_curve(
            BOOST_L2_LIST,
            [d["r2_mean"] for d in l2_curve],
            "l2_regularization",
            f"BOOST tuning @ FPS={tune_fps}",
            OUT_DIR / f"boost_tune_l2_regularization_fps{tune_fps}.png",
            xscale="log" if max(BOOST_L2_LIST) > 0 else None,
        )

        base_b5 = dict(base_b4)
        base_b5["l2_regularization"] = best_l2
        subs_curve = boost_param_sweep(X, Y, groups, base_b5, "subsample", BOOST_SUBSAMPLE_LIST, BOOST_TUNE_MAX_ITER)
        results[tune_fps]["models"]["boost_tuning"]["subsample"] = subs_curve
        best_sub = max(subs_curve, key=lambda d: d["r2_mean"])["subsample"]
        plot_1d_curve(
            BOOST_SUBSAMPLE_LIST,
            [d["r2_mean"] for d in subs_curve],
            "subsample",
            f"BOOST tuning @ FPS={tune_fps}",
            OUT_DIR / f"boost_tune_subsample_fps{tune_fps}.png",
        )

        if BOOST_TUNE_MAX_ITER_FINAL:
            base_b6 = dict(base_b5)
            base_b6["subsample"] = best_sub

            maxiter_curve = []
            for mi in BOOST_MAXITER_LIST:
                params = dict(base_b6)
                params["max_iter"] = mi
                stats = cv_r2_rmse(MultiOutputRegressor(HistGradientBoostingRegressor(**params)), X, Y, groups, N_SPLITS)
                maxiter_curve.append({"max_iter": mi, **stats})

            results[tune_fps]["models"]["boost_tuning"]["max_iter"] = maxiter_curve
            plot_1d_curve(
                BOOST_MAXITER_LIST,
                [d["r2_mean"] for d in maxiter_curve],
                "max_iter",
                f"BOOST tuning @ FPS={tune_fps}",
                OUT_DIR / f"boost_tune_max_iter_fps{tune_fps}.png",
                xscale="log",
            )

    out_json = OUT_DIR / "fps_sweep_results.json"
    out_json.write_text(json.dumps(results, indent=2))

    fps_vals = list(FPS_LIST)
    plt.figure()

    if RUN_RIDGE:
        plt.plot(
            fps_vals,
            [results[f]["models"]["ridge"]["best_r2_mean"] for f in FPS_LIST],
            marker="o",
            label="Ridge",
        )
    if RUN_RF:
        plt.plot(
            fps_vals,
            [results[f]["models"]["rf"]["r2_mean"] for f in FPS_LIST],
            marker="o",
            label="Random Forest",
        )
    if RUN_BOOST:
        plt.plot(
            fps_vals,
            [results[f]["models"]["boost"]["r2_mean"] for f in FPS_LIST],
            marker="o",
            label="Boosting",
        )

    plt.xlabel("FPS")
    plt.ylabel("CV R2 (mean)")
    plt.title(f"FPS sweep | Y_TARGET='{Y_TARGET}'")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fps_vs_r2.png", dpi=200)

    if RUN_RIDGE:
        best_fps = max(FPS_LIST, key=lambda f: results[f]["models"]["ridge"]["best_r2_mean"])
        curve = results[best_fps]["models"]["ridge"]["curve"]
        alphas = [c["alpha"] for c in curve]
        r2s = [c["r2_mean"] for c in curve]

        plt.figure()
        plt.semilogx(alphas, r2s, marker="o")
        plt.xlabel("Ridge alpha (log)")
        plt.ylabel("CV R2 (mean)")
        plt.title(f"Ridge alpha sweep @ FPS={best_fps}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"ridge_alpha_curve_fps{best_fps}.png", dpi=200)


if __name__ == "__main__":
    main()