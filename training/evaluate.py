import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

import seaborn as sns
import librosa



BASE_DIR = Path(__file__).resolve().parent.parent

DATASET = BASE_DIR / "dataset" / "processed"

MODEL_PATH = BASE_DIR / "models" / "bird_model.keras"

OUTPUT_DIR = BASE_DIR / "evaluation"

SAMPLE_RATE = 16000
IMG_SIZE = 128


def audio_to_spectrogram(path):

    audio, sr = librosa.load( path, sr=SAMPLE_RATE )


    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=IMG_SIZE
    )

    mel = librosa.power_to_db( mel, ref=np.max )

    mel = tf.image.resize( mel[..., None], [IMG_SIZE, IMG_SIZE] )

    return mel.numpy()



def load_test_data():

    X = []
    y = []

    classes = sorted(
        [
            p.name
            for p in DATASET.iterdir()
            if p.is_dir()
        ]
    )


    class_map = {
        c:i
        for i,c in enumerate(classes)
    }


    for species in classes:

        files = list(
            (DATASET/species).glob("*.wav")
        )

        for file in files:

            X.append(
                audio_to_spectrogram(file)
            )

            y.append(
                class_map[species]
            )


    return (
        np.array(X),
        np.array(y),
        classes
    )



def plot_history():

    history_file = BASE_DIR / "models" / "history.npy"

    if not history_file.exists():
        print(
            "No training history found"
        )
        return


    history = np.load(
        history_file,
        allow_pickle=True
    ).item()


    plt.figure()

    plt.plot( history["accuracy"] )

    plt.plot( history["val_accuracy"] )

    plt.title( "Model Accuracy" )

    plt.xlabel( "Epoch" )

    plt.ylabel( "Accuracy" )

    plt.legend(
        [
            "Training",
            "Validation"
        ]
    )

    plt.savefig( OUTPUT_DIR/"accuracy.png" )

    plt.close()



def main():

    OUTPUT_DIR.mkdir( exist_ok=True )


    print( "Loading model..." )

    model = tf.keras.models.load_model( MODEL_PATH )

    print( "Loading dataset..." )

    X,y,classes = load_test_data()

    print( "Predicting..." )

    predictions = model.predict( X )

    predicted = np.argmax( predictions, axis=1 )


    # -----------------------
    # Classification report
    # -----------------------

    report = classification_report(
        y,
        predicted,
        target_names=classes
    )


    print(report)


    with open(
        OUTPUT_DIR/"report.txt",
        "w"
    ) as f:
        f.write(report)



    # -----------------------
    # Confusion matrix
    # -----------------------

    cm = confusion_matrix( y, predicted )

    plt.figure( figsize=(12,10) )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=classes,
        yticklabels=classes
    )

    plt.title( "Confusion Matrix" )

    plt.xlabel( "Predicted" )

    plt.ylabel( "Actual" )

    plt.xticks( rotation=90 )

    plt.tight_layout()

    plt.savefig( OUTPUT_DIR/"confusion_matrix.png" )

    plt.close()

    print( "Saved results to", OUTPUT_DIR )



if __name__ == "__main__":
    main()