"""MobileNetV3-Small frozen-backbone model and training helpers."""

from pathlib import Path


def build_frozen_model(image_size, num_classes, dropout_rate=0.2):
    """Build an ImageNet-pretrained MobileNetV3-Small with a frozen backbone."""
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(*image_size, 3), name="image")
    backbone = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        weights="imagenet",
        input_shape=(*image_size, 3),
        include_preprocessing=True,
    )
    backbone.trainable = False
    features = backbone(inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(features)
    features = tf.keras.layers.Dropout(dropout_rate, name="classifier_dropout")(features)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="predictions"
    )(features)
    return tf.keras.Model(inputs, outputs, name="mobilenet_v3_small_frozen")


def train_frozen_model(
    model,
    train_dataset,
    validation_dataset,
    epochs,
    learning_rate,
    patience,
    checkpoint_path,
):
    """Train only the classification head and restore the best validation model."""
    import tensorflow as tf

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    return model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )
