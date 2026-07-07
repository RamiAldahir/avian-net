import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path

import tensorflow as tf
import librosa
import numpy as np

from model import build_model


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET = BASE_DIR / "dataset" / "processed"
MODEL_DIR = BASE_DIR / "models"

SAMPLE_RATE = 16000

IMG_SIZE = 128



def audio_to_spectrogram(path):

    audio, sr = librosa.load(
        path,
        sr=SAMPLE_RATE
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=IMG_SIZE
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # resize to 128x128

    mel_db = tf.image.resize(
        mel_db[..., np.newaxis],
        [IMG_SIZE, IMG_SIZE]
    )

    return mel_db.numpy()


def load_dataset():

    X = []
    y = []

    classes = sorted(
        [
            x.name
            for x in DATASET.iterdir()
            if x.is_dir()
        ]
    )

    class_map = {
        name:i
        for i,name in enumerate(classes)
    }

    print(classes)

    for species in classes:

        folder = DATASET / species

        for file in folder.glob("*.wav"):

            try:
                spec = audio_to_spectrogram(file)

                X.append(spec)
                y.append(class_map[species])

            except Exception as e:
                print( "Failed:", file, e )

    return (
        np.array(X),
        np.array(y),
        classes
    )


def main():

    X,y,classes = load_dataset()

    print( "Dataset:", X.shape )

    model = build_model(
        len(classes)
    )

    history = model.fit(
        X,
        y,
        epochs=10,
        validation_split=0.2,
        batch_size=32
    )

    np.save( MODEL_DIR / "history.npy", history.history )

    model.save( MODEL_DIR / "bird_model.keras" )

    converter = tf.lite.TFLiteConverter.from_keras_model( model )

    tflite_model = converter.convert()


    with open(
        MODEL_DIR / "bird_model.tflite",
        "wb"
    ) as f:
        f.write(tflite_model)


    with open(
        MODEL_DIR / "labels.txt",
        "w"
    ) as f:
        for c in classes:
            f.write(c+"\n")



if __name__ == "__main__":
    main()