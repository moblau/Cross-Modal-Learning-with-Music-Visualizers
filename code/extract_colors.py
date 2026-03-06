import numpy as np
import cv2
from pathlib import Path

PROCESSED_DIR = Path("videos/mvt_processed")
OUTPUT_NPZ = Path("mvt_colors_dataset_8fps.npz")

TARGET_FPS = 8
H_BINS = 12
EPS = 1e-9


def hue_hist(frame, bins):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]
    hist = cv2.calcHist([h], [0], None, [bins], [0, 180]).ravel().astype(np.float32)
    return hist / (hist.sum() + EPS)


def weight_hue(hist):
    b = hist.shape[0]
    theta = (np.arange(b, dtype=np.float32) + 0.5) * (2.0 * np.pi / b)
    return float(hist.max()), float((hist * np.sin(theta)).sum()), float((hist * np.cos(theta)).sum())


def mean_v(frame):
    v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32) / 255.0
    return float(v.mean())


def mean_lab(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    return lab.reshape(-1, 3).mean(axis=0).astype(np.float32)


def motion(prev, cur):
    if prev is None:
        return 0.0
    return float((cv2.absdiff(prev, cur).astype(np.float32) / 255.0).mean())


def sample_frames(vp, fps):
    cap = cv2.VideoCapture(str(vp))
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step, next_t, idx, fidx = 1.0 / fps, 0.0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fidx / src
        if t + 1e-9 >= next_t:
            yield idx, next_t, frame
            idx += 1
            next_t += step
        fidx += 1
    cap.release()


def main():
    vids = sorted(PROCESSED_DIR.glob("*_proxy.mp4"))
    Ys, Ts, names = [], [], []

    for vp in vids:
        y_list, t_list = [], []
        prev = None

        for _, t, frame in sample_frames(vp, TARGET_FPS):
            H = hue_hist(frame, H_BINS)
            w, hs, hc = weight_hue(H)
            b = mean_v(frame)
            lab = mean_lab(frame)
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            m = motion(prev, g)
            prev = g

            y_list.append([w, hs, hc, b, lab[0], lab[1], lab[2], m])
            t_list.append(t)

        Y = np.asarray(y_list, np.float32)
        T = np.asarray(t_list, np.float32)
        Ys.append(Y); Ts.append(T)
        names.append(vp.stem.replace("_proxy", ""))

    lengths = np.array([y.shape[0] for y in Ys], np.int32)
    V, max_T = len(Ys), int(lengths.max()) if len(Ys) else 0

    Y_pad = np.full((V, max_T, 8), np.nan, np.float32)
    T_pad = np.full((V, max_T), np.nan, np.float32)

    for i, (Y, T) in enumerate(zip(Ys, Ts)):
        n = Y.shape[0]
        Y_pad[i, :n] = Y
        T_pad[i, :n] = T

    np.savez_compressed(OUTPUT_NPZ, Y=Y_pad, times=T_pad, lengths=lengths, names=np.array(names, dtype=object))


if __name__ == "__main__":
    main()