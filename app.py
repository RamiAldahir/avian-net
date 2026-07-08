import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import librosa
import numpy as np
import sounddevice as sd

from pathlib import Path

import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES

BASE_DIR = Path(__file__).resolve().parent

MODEL = BASE_DIR / "models" / "bird_model.tflite"
LABELS = BASE_DIR / "models" / "labels.txt"

RATE = 16000
SECONDS = 5

interpreter = tf.lite.Interpreter(
    model_path=str(MODEL)
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



with open(LABELS) as f:
    labels = [
        x.strip()
        for x in f.readlines()
    ]


def make_spec(audio):

    mel = librosa.feature.melspectrogram( y=audio, sr=RATE, n_mels=128 )
    mel = librosa.power_to_db( mel, ref=np.max )
    mel = tf.image.resize( mel[...,None], [128,128] )

    return mel.numpy()


def predict(audio):

    spec = make_spec(audio)
    spec = np.expand_dims( spec, axis=0 )

    interpreter.set_tensor( input_details[0]["index"], spec.astype(np.float32) )
    interpreter.invoke()
    output = interpreter.get_tensor( output_details[0]["index"] )[0]

    index = np.argmax(output)

    return ( labels[index], output[index] )


# while True:

#     print("Listening...")

#     recording = sd.rec(
#         int(RATE*SECONDS),
#         samplerate=RATE,
#         channels=1
#     )

#     sd.wait()

#     audio = recording.flatten()
#     bird, confidence = predict( audio )

#     print( bird, f"{confidence:.2%}" )

# Using tkinter to make a simple drag-anddrop GUI for testing with sound files
def on_drop(event):
    try:
        # Handles filenames with spaces
        file_path = event.widget.tk.splitlist(event.data)[0]

        # Remove file:// if present
        if file_path.startswith("file://"):
            file_path = file_path[7:]

        file_path = os.path.normpath(file_path)

        print(f"File: {file_path}")

        if not os.path.isfile(file_path):
            print("File does not exist.")
            return

        if not file_path.lower().endswith(".mp3") :
            print("MP3 Files Only!")
            return

        print("Loading audio...")

        audio, _ = librosa.load(
            file_path,
            sr=RATE,
            mono=True
        )

        expected_samples = RATE * SECONDS

        if len(audio) < expected_samples:
            audio = np.pad(audio, (0, expected_samples - len(audio)))
        else:
            audio = audio[:expected_samples]

        bird, confidence = predict(audio)

        print(f"Prediction: {bird} ({confidence:.2%})")

        label.config(
            text=f"{os.path.basename(file_path)}\n\n{bird}\n{confidence:.2%}",
            bg="lightgreen"
        )

    except Exception as e:
        print("Error:", e)
        label.config(text=str(e), bg="red")


root = TkinterDnD.Tk()
root.title("Bird Classifier")
root.geometry("500x300")

label = tk.Label(
    root,
    text="Drag and drop an MP3 file here",
    font=("Arial", 14),
    relief="ridge",
    bd=3,
    bg="white"
)

label.pack(fill="both", expand=True, padx=20, pady=20)

# Enable dropping files
label.drop_target_register(DND_FILES)

label.dnd_bind("<<Drop>>", on_drop)
label.dnd_bind("<<DragEnter>>", lambda e: label.config(bg="lightgreen"))
label.dnd_bind("<<DragLeave>>", lambda e: label.config(bg="white"))

root.mainloop()