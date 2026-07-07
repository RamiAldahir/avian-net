import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "dataset" / "recordings"
OUTPUT_DIR = BASE_DIR / "dataset" / "processed"

TARGET_SR = 16000
CLIP_LENGTH = 5  # seconds
MIN_RMS = 0.005

# -----------------------------


def split_audio(audio, sr):
    clip_samples = sr * CLIP_LENGTH

    for start in range(0, len(audio), clip_samples):
        clip = audio[start:start + clip_samples]

        if len(clip) < clip_samples:
            continue

        yield clip


def is_silent(clip):
    rms = np.sqrt(np.mean(np.square(clip)))
    return rms < MIN_RMS


def process_file(file_path, output_folder):

    audio, sr = librosa.load(
        file_path,
        sr=TARGET_SR,
        mono=True,
    )

    stem = file_path.stem

    clip_number = 0

    for clip in split_audio(audio, TARGET_SR):

        if is_silent(clip):
            continue

        out = output_folder / f"{stem}_{clip_number:03d}.wav"

        sf.write(
            out,
            clip,
            TARGET_SR,
            subtype="PCM_16"
        )

        clip_number += 1


def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    species = [p for p in INPUT_DIR.iterdir() if p.is_dir()]

    for species_folder in species:

        print(species_folder.name)

        out_folder = OUTPUT_DIR / species_folder.name
        out_folder.mkdir(exist_ok=True)

        files = list(species_folder.glob("*.mp3"))

        for file in tqdm(files):

            try:
                process_file(file, out_folder)

            except Exception as e:
                print(file.name, e)


if __name__ == "__main__":
    main()