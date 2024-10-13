import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

def main():
    # 1. Caricamento del dataset MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 2. Normalizzazione dei dati
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # 3. Aggiunta della dimensione del canale
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # 4. Costruzione del modello
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # 5. Compilazione del modello
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 6. Addestramento del modello
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.1
    )

    # 7. Valutazione del modello
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_accuracy:.4f}")

    # 8. Salvataggio del modello
    model_save_path = 'mnist_model.h5'
    model.save(model_save_path)
    print(f"Modello salvato in: {os.path.abspath(model_save_path)}")

    # 9. Caricamento del modello salvato
    loaded_model = keras.models.load_model(model_save_path)
    print("Modello caricato con successo.")

    # 10. Fare previsioni con il modello caricato
    predictions = loaded_model.predict(x_test)

    # Mostra alcune previsioni
    num_predictions = 5
    for i in range(num_predictions):
        print(f"Predizione per l'immagine {i}: {np.argmax(predictions[i])}, Etichetta reale: {y_test[i]}")

if __name__ == "__main__":
    main()
