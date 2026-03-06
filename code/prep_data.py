import subprocess
from pathlib import Path

IN_DIR = Path("videos/nature")
OUT_DIR = Path("videos/nature_processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FFMPEG = "/opt/homebrew/bin/ffmpeg"
AUDIO_SR = 22050
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi"}


def run(cmd):
    subprocess.run(cmd, check=True)


def main():
    vids = sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS])
    for p in vids:
        stem = p.stem
        proxy = OUT_DIR / f"{stem}_proxy.mp4"
        wav = OUT_DIR / f"{stem}_audio.wav"
        if proxy.exists() and wav.exists():
            continue

        if not proxy.exists():
            vf = "scale=320:180:force_original_aspect_ratio=decrease,pad=320:180:(ow-iw)/2:(oh-ih)/2,fps=30"
            run([FFMPEG, "-y", "-i", str(p), "-vf", vf, "-an",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
                 "-movflags", "+faststart", str(proxy)])

        if not wav.exists():
            run([FFMPEG, "-y", "-i", str(p), "-vn", "-ac", "1", "-ar", str(AUDIO_SR),
                 "-c:a", "pcm_s16le", str(wav)])

    print("done")


if __name__ == "__main__":
    main()