import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("songs")
OUTPUT_NPZ = Path("null_point.npz")

TARGET_FPS = 8
N_MELS = 64
N_FFT = 2048
HOP_LENGTH = 512


def mean_bins(feat, t, n_bins):
    D = feat.shape[0]
    out = np.full((n_bins, D), np.nan, np.float32)
    b = np.clip((t * TARGET_FPS).astype(int), 0, n_bins - 1)
    for i in range(n_bins):
        m = (b == i)
        if np.any(m):
            out[i] = feat[:, m].mean(axis=1).astype(np.float32)
    return out


def delta(x):
    x = np.asarray(x, np.float32).reshape(-1)
    d = np.zeros_like(x)
    d[1:] = x[1:] - x[:-1]
    return d


def extract(wav):
    import librosa

    y, sr = librosa.load(str(wav), sr=None, mono=True)
    n_bins = max(1, int(np.ceil((len(y) / sr) * TARGET_FPS)))

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)
    mel = librosa.power_to_db(S, ref=np.max)
    n_frames = mel.shape[1]
    t = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=HOP_LENGTH)

    mag = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    rms = librosa.feature.rms(S=mag)
    centroid = librosa.feature.spectral_centroid(S=mag, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(S=mag, sr=sr, roll_percent=0.85)
    bandwidth = librosa.feature.spectral_bandwidth(S=mag, sr=sr)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH).reshape(1, -1)

    mel = mean_bins(mel, t, n_bins)
    rms = mean_bins(rms, t, n_bins)[:, 0]
    centroid = mean_bins(centroid, t, n_bins)[:, 0]
    rolloff = mean_bins(rolloff, t, n_bins)[:, 0]
    bandwidth = mean_bins(bandwidth, t, n_bins)[:, 0]
    onset = mean_bins(onset, t, n_bins)[:, 0]

    low = np.nanmean(mel[:, :6], axis=1).astype(np.float32)
    mid = np.nanmean(mel[:, 6:26], axis=1).astype(np.float32)
    high = np.nanmean(mel[:, 26:], axis=1).astype(np.float32)

    X = np.column_stack([
        mel.astype(np.float32),
        rms, centroid, rolloff, bandwidth, onset,
        low, mid, high,
        delta(rms), delta(centroid), delta(low), delta(mid), delta(high)
    ]).astype(np.float32)

    times = (np.arange(n_bins, dtype=np.float32) / TARGET_FPS).astype(np.float32)
    return X, times


def main():
    wavs = sorted(PROCESSED_DIR.glob("*.wav"))
    Xs, Ts, names = [], [], []

    for wp in wavs:
        X, t = extract(wp)
        Xs.append(X); Ts.append(t); names.append(wp.stem)

    lengths = np.array([x.shape[0] for x in Xs], np.int32)
    V, max_T = len(Xs), int(lengths.max()) if len(Xs) else 0
    D = int(Xs[0].shape[1]) if len(Xs) else 0

    X_all = np.full((V, max_T, D), np.nan, np.float32)
    T_all = np.full((V, max_T), np.nan, np.float32)
    for i, (X, t) in enumerate(zip(Xs, Ts)):
        n = X.shape[0]
        X_all[i, :n] = X
        T_all[i, :n] = t

    np.savez_compressed(OUTPUT_NPZ, X=X_all, times=T_all, lengths=lengths, names=np.array(names, dtype=object))


if __name__ == "__main__":
    main()