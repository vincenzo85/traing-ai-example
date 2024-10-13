import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
import sys
import os

def load_and_preprocess_image(image_path):
    """
    Carica un'immagine, la converte in scala di grigi, la ridimensiona a 28x28,
    inverte i colori (se necessario), normalizza i pixel e la prepara per la previsione.
    """
    try:
        # Apri l'immagine
        img = Image.open(image_path).convert('L')  # Converti in scala di grigi
    except Exception as e:
        print(f"Errore nell'aprire l'immagine: {e}")
        sys.exit(1)
    
    # Ridimensiona l'immagine a 28x28
    img = img.resize((28, 28), Image.LANCZOS)
    
    # Inverti i colori se lo sfondo è bianco
    # MNIST ha lo sfondo nero e il numero bianco
    # Se il numero è scuro, inverti i colori
    img_array = np.array(img)
    if np.mean(img_array) > 127:
        img = ImageOps.invert(img)
        img_array = np.array(img)
    
    # Normalizza i pixel
    img_array = img_array.astype('float32') / 255.0
    
    # Aggiungi dimensioni per il canale e il batch
    img_array = np.expand_dims(img_array, axis=-1)  # Shape: (28, 28, 1)
    img_array = np.expand_dims(img_array, axis=0)   # Shape: (1, 28, 28, 1)
    
    return img_array

def main():
    if len(sys.argv) != 2:
        print("Uso corretto: python predict_number.py <percorso_immagine>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Errore: Il file '{image_path}' non esiste.")
        sys.exit(1)
    
    # Carica e preprocessa l'immagine
    img = load_and_preprocess_image(image_path)
    
    # Carica il modello salvato
    model_path = 'mnist_model.h5'
    if not os.path.exists(model_path):
        print(f"Errore: Il modello '{model_path}' non è stato trovato. Assicurati di aver eseguito 'mnist_model.py' prima.")
        sys.exit(1)
    
    model = keras.models.load_model(model_path)
    print("Modello caricato con successo.")
    
    # Effettua la previsione
    predictions = model.predict(img)
    predicted_digit = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    print(f"Il numero previsto è: {predicted_digit} con una confidenza di {confidence*100:.2f}%")

if __name__ == "__main__":
    main()
