import csv, re
from pathlib import Path

import joblib
import numpy as np


AUDIO_NPZ = Path("null_point.npz")
MODEL_JOBLIB = Path("mvt_boost_model.joblib")   # or Path("mvt_rf_model.joblib")
OUT_DIR = Path("td_controls_csv")

TARGET_FPS = None          # None: use npz["times"] if present
SMOOTH_ALPHA = 0.0         # 0.0 = off
HEADER = ("t", "weight", "motion")


def sanitize(s: str) -> str:
    return re.sub(r"[^\w\-\.]+", "_", str(s).strip())[:180]


def ema(x: np.ndarray, a: float) -> np.ndarray:
    if a <= 0.0 or len(x) == 0:
        return x
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * y[i - 1] + (1.0 - a) * x[i]
    return y


def fill_nan(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    for k in range(y.shape[1]):
        c = y[:, k]
        last = np.nan
        for i in range(len(c)):
            if np.isfinite(c[i]): last = c[i]
            elif np.isfinite(last): c[i] = last
        if not np.isfinite(c[0]):
            idx = np.flatnonzero(np.isfinite(c))
            if len(idx): c[:idx[0]] = c[idx[0]]
        y[:, k] = c
    return y


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    blob = joblib.load(MODEL_JOBLIB)
    model = blob["model"] if isinstance(blob, dict) and "model" in blob else blob

    a = np.load(AUDIO_NPZ, allow_pickle=True)
    X = a["X"]
    lengths = a["lengths"].astype(int)
    names = [str(n) for n in a["names"].tolist()]
    times = a["times"] if "times" in a.files else None

    for i, name in enumerate(names):
        n = int(lengths[i])
        if n <= 0:
            continue

        Xi = X[i, :n, :].astype(np.float32)
        good = np.isfinite(Xi).all(axis=1)
        if not np.any(good):
            continue

        pred = np.full((n, 2), np.nan, dtype=np.float32)
        pred[good] = np.asarray(model.predict(Xi[good]), dtype=np.float32)

        if SMOOTH_ALPHA > 0.0:
            pred = ema(fill_nan(pred), SMOOTH_ALPHA)

        if times is not None and TARGET_FPS is None:
            t = times[i, :n].astype(np.float32)
        else:
            fps = float(TARGET_FPS) if TARGET_FPS is not None else 8.0
            t = (np.arange(n, dtype=np.float32) / fps).astype(np.float32)

        out = OUT_DIR / f"{sanitize(name)}_controls.csv"
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            for ti, (wgt, mot) in zip(t, pred):
                w.writerow([float(ti),
                            "" if not np.isfinite(wgt) else float(wgt),
                            "" if not np.isfinite(mot) else float(mot)])


if __name__ == "__main__":
    main()