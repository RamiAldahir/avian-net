import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score
)

import seaborn as sns
import librosa

plt.style.use("default")

sns.set_theme(
    style="white",
    palette="Blues"
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET = BASE_DIR / "dataset" / "processed"

MODEL_PATH = BASE_DIR / "models" / "bird_model.keras"

OUTPUT_DIR = BASE_DIR / "evaluation"

SAMPLE_RATE = 16000
IMG_SIZE = 128

SCATTER_SIZE = 15


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
        files = list( (DATASET/species).glob("*.wav") )
        for file in files:
            X.append( audio_to_spectrogram(file) )
            y.append( class_map[species] )

    return (
        np.array(X),
        np.array(y),
        classes
    )

def plot_history():

    history_file = BASE_DIR / "models" / "history.npy"

    if not history_file.exists():
        print("No training history found")
        return

    history = np.load( history_file, allow_pickle=True ).item()
    epochs = np.arange( 1, len(history["accuracy"]) + 1 )

    def scatter_metric(key, title, filename):

        if key not in history:
            return
        plt.figure(figsize=(8,5))
        plt.scatter( epochs, history[key], s=SCATTER_SIZE )
        plt.plot( epochs, history[key], alpha=0.5 )
        plt.title(title)
        plt.xlabel( "Epoch" )
        plt.ylabel( "Score" )
        plt.ylim( 0, 1 )
        plt.grid( alpha=0.3 )
        plt.tight_layout()
        plt.savefig( OUTPUT_DIR / filename, dpi=300 )
        plt.close()

    # Accuracy
    plt.figure(figsize=(8,5))
    plt.scatter( epochs, history["accuracy"], label="Training", s=SCATTER_SIZE )
    plt.scatter( epochs, history["val_accuracy"], label="Validation", s=SCATTER_SIZE )
    plt.plot( epochs, history["accuracy"], label="Training", alpha=0.5 )
    plt.plot( epochs, history["val_accuracy"], label="Validation", alpha=0.5 )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title( "Accuracy" )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig( OUTPUT_DIR/"accuracy.png", dpi=300 )
    plt.close()

    # Loss
    if "loss" in history:
        plt.figure(figsize=(8,5))
        plt.scatter( epochs, history["loss"], label="Training", s=SCATTER_SIZE )
        plt.scatter( epochs, history["val_loss"], label="Validation", s=SCATTER_SIZE )
        plt.plot( epochs, history["loss"], label="Training", alpha=0.5 )
        plt.plot( epochs, history["val_loss"], label="Validation", alpha=0.5 )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title( "Loss" )
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig( OUTPUT_DIR/"loss.png", dpi=300 )
        plt.close()

    # Metrics
    scatter_metric( "val_precision", "Validation Precision", "precision.png" )
    scatter_metric( "val_recall", "Validation Recalls", "recall.png" )
    scatter_metric( "val_f1", "Validation F1 Score", "f1.png" )



def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_history()

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

    report = classification_report( y, predicted, target_names=classes, digits=4 )
    accuracy = accuracy_score(y, predicted)
    # precision, recall, f1, support = precision_recall_fscore_support( y, predicted, zero_division=0 )
    print(f"\nOverall Accuracy: {accuracy:.4f}\n")
    print(report)
    with open( OUTPUT_DIR/"report.txt", "w" ) as f: f.write(report)

    # -----------------------
    # Confusion matrix
    # -----------------------

    cm = confusion_matrix(y, predicted)

    plt.figure(figsize=(12,10))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig( OUTPUT_DIR / "confusion_matrix.png", dpi=300 )
    plt.close()


if __name__ == "__main__":
    main()