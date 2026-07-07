import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import librosa
import numpy as np
import sounddevice as sd

from pathlib import Path

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



while True:

    print("Listening...")

    recording = sd.rec(
        int(RATE*SECONDS),
        samplerate=RATE,
        channels=1
    )

    sd.wait()

    audio = recording.flatten()
    bird, confidence = predict( audio )

    print( bird, f"{confidence:.2%}" )