import tensorflow as tf

def build_model(image_size):

    model = tf.keras.models.Sequential([

        # Adding input layer 
        tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3)), 

        # Adding alternating convolutional and pooling layers 
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 
        tf.keras.layers.MaxPooling2D(2, 2),

        # Adding flattening layer 
        tf.keras.layers.Flatten(),

        # Adding hidden layer 
        tf.keras.layers.Dense(64, activation='relu'), 

        # Adding dropout to reduce overfitting
        tf.keras.layers.Dropout(0.5),

        # Adding output layer; 5 represents the 5 classes
        tf.keras.layers.Dense(5, activation='softmax'), 
    ])

    return model

def train_model(model, train_generator, val_generator, num_epochs, best_model_file):

    # Compiling the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Configuring early stopping 
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,               # 10 epoch buffer
        restore_best_weights=True  
    )

    # Saving the best model 
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath= best_model_file, 
        monitor='val_loss',       
        save_best_only=True,      
    )

    # Passing both callbacks into training loop, and fitting training data to model 
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=num_epochs,
        callbacks=[early_stopping, model_checkpoint] 
    )  

    return history 



