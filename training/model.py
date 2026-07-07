import tensorflow as tf


def build_model(num_classes):

    model = tf.keras.Sequential([
        
        tf.keras.layers.Input( shape=(128, 128, 1) ),

        tf.keras.layers.Conv2D( 32, (3,3), activation="relu" ),
        
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D( 64, (3,3), activation="relu" ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D( 128, (3,3), activation="relu" ),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense( 256, activation="relu" ),

        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense( num_classes, activation="softmax" )
    ])


    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model