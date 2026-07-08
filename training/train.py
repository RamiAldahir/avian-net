import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path

import tensorflow as tf
import librosa
import numpy as np

from model import build_model
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET = BASE_DIR / "dataset" / "processed"
MODEL_DIR = BASE_DIR / "models"

SAMPLE_RATE = 16000

IMG_SIZE = 128

EPOCHS = 30



def audio_to_spectrogram(path):

    audio, sr = librosa.load( path, sr=SAMPLE_RATE )
    mel = librosa.feature.melspectrogram( y=audio, sr=sr, n_mels=IMG_SIZE )
    mel_db = librosa.power_to_db( mel, ref=np.max )
    # resize to 128x128
    mel_db = tf.image.resize( mel_db[..., np.newaxis], [IMG_SIZE, IMG_SIZE] )

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

class MetricsCallback(tf.keras.callbacks.Callback):

    def __init__(self, validation_data):
        super().__init__()

        self.validation_data = validation_data

        self.val_precision = []
        self.val_recall = []
        self.val_f1 = []


    def on_epoch_end(self, epoch, logs=None):

        X_val, y_val = self.validation_data

        predictions = self.model.predict( X_val, verbose=0 )

        predicted_classes = np.argmax( predictions, axis=1 )

        precision = precision_score( y_val, predicted_classes, average="macro", zero_division=0 )

        recall = recall_score( y_val, predicted_classes, average="macro", zero_division=0 )

        f1 = f1_score( y_val, predicted_classes, average="macro", zero_division=0 )

        self.val_precision.append(precision)
        self.val_recall.append(recall)
        self.val_f1.append(f1)

        print(
            f" - val_precision: {precision:.4f}"
            f" - val_recall: {recall:.4f}"
            f" - val_f1: {f1:.4f}"
        )

def main():

    X,y,classes = load_dataset()

    print( "Dataset:", X.shape )

    model = build_model(
        len(classes)
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    metric_callback = MetricsCallback(
        (
            X_val,
            y_val
        )
    )

    history = model.fit(
        X_train,
        y_train,
        epochs= EPOCHS,
        validation_data=(
            X_val,
            y_val
        ),
        batch_size=16,
        callbacks=[
            metric_callback
        ]
    )

    history_data = history.history

    history_data["val_precision"] = ( metric_callback.val_precision )
    history_data["val_recall"] = ( metric_callback.val_recall )
    history_data["val_f1"] = ( metric_callback.val_f1 )

    np.save( MODEL_DIR / "history.npy", history_data )

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