# Riconoscimento dei Numeri Scritti a Mano con TensorFlow e Keras

Questo progetto implementa un semplice sistema di riconoscimento dei numeri scritti a mano utilizzando una rete neurale in Python con le librerie **TensorFlow** e **Keras**. Il sistema è composto da due script principali:

1. **`cifra.py`**: Script per addestrare e salvare il modello di riconoscimento dei numeri.
2. **`predict_number.py`**: Script per caricare il modello addestrato e prevedere il numero in un'immagine fornita.

## Indice

- [Requisiti](#requisiti)
- [Installazione](#installazione)
- [Struttura del Progetto](#struttura-del-progetto)
- [Utilizzo](#utilizzo)
  - [Addestramento del Modello (`cifra.py`)](#addestramento-del-modello-cifrapy)
  - [Previsione di un Numero (`predict_number.py`)](#previsione-di-un-numero-predict_numberpy)
- [Dettagli del Funzionamento](#dettagli-del-funzionamento)
- [Risoluzione dei Problemi](#risoluzione-dei-problemi)
- [Ulteriori Miglioramenti](#ulteriori-miglioramenti)
- [Risorse Utili](#risorse-utili)

## Requisiti

Assicurati di avere i seguenti requisiti prima di procedere:

- **Python 3.6 o superiore**: Puoi verificarlo eseguendo `python3 --version` nel terminale.
- **Pip**: Gestore di pacchetti per Python.
- **Virtualenv (opzionale ma consigliato)**: Per creare ambienti virtuali isolati.

## Installazione

Segui questi passaggi per configurare l'ambiente di sviluppo e installare le dipendenze necessarie.

### 1. Clona il Repository (Se Applicabile)

Se il progetto è su GitHub o un altro repository, clonalo. Altrimenti, assicurati di avere gli script `cifra.py` e `predict_number.py` nella tua directory di lavoro.

```bash
git clone https://github.com/tuo-utente/tuo-repo.git
cd tuo-repo
```

### 2. Crea un Ambiente Virtuale (Consigliato)

Creare un ambiente virtuale aiuta a isolare le dipendenze del progetto.

```bash
python3 -m venv venv
```

Attiva l'ambiente virtuale:

- **Su Linux/MacOS:**

  ```bash
  source venv/bin/activate
  ```

- **Su Windows:**

  ```bash
  venv\Scripts\activate
  ```

### 3. Aggiorna `pip`

Assicurati di avere l'ultima versione di `pip`.

```bash
pip install --upgrade pip
```

### 4. Installa le Dipendenze

Installa le librerie necessarie usando `pip`.

```bash
pip install tensorflow pillow numpy matplotlib
```

> **Nota:** Se stai utilizzando una GPU e desideri sfruttarla per accelerare l'addestramento e l'inferenza, segui la [Guida Ufficiale di TensorFlow per l'Installazione con GPU](https://www.tensorflow.org/install/gpu) e installa le versioni appropriate di CUDA e cuDNN.

## Struttura del Progetto

La struttura del progetto dovrebbe essere simile a questa:

```
tuo-progetto/
│
├── cifra.py
├── predict_number.py
├── mnist_model.h5
├── README.md
└── venv/ (cartella ambiente virtuale)
```

## Utilizzo

### Addestramento del Modello (`cifra.py`)

Questo script carica il dataset MNIST, addestra una rete neurale semplice e salva il modello addestrato.

#### Esecuzione dello Script

1. **Assicurati di essere nell'ambiente virtuale:**

   ```bash
   source venv/bin/activate  # Su Linux/MacOS
   # oppure
   venv\Scripts\activate     # Su Windows
   ```

2. **Esegui lo script:**

   ```bash
   python3 cifra.py
   ```

#### Output Atteso

Durante l'esecuzione, vedrai l'addestramento del modello con l'accuratezza e la perdita che cambiano ad ogni epoca. Alla fine, lo script salverà il modello addestrato in un file chiamato `mnist_model.h5` e mostrerà alcune previsioni sul set di test.

Esempio di output:

```
x_train shape: (60000, 28, 28, 1)
y_train shape: (60000,)
x_test shape: (10000, 28, 28, 1)
y_test shape: (10000,)
Epoch 1/5
1688/1688 [==============================] - 10s 6ms/step - loss: 0.3274 - accuracy: 0.8976 - val_loss: 0.1605 - val_accuracy: 0.9521
...
Test accuracy: 0.9800
Modello salvato in: /percorso/assoluto/mnist_model.h5
Modello caricato con successo.
Predizione per l'immagine 0: 7, Etichetta reale: 7
...
```

### Previsione di un Numero (`predict_number.py`)

Questo script carica un'immagine di un numero, preprocessa l'immagine, carica il modello addestrato e prevede il numero presente nell'immagine.

#### Preparazione dell'Immagine

- **Formato:** PNG o JPG.
- **Dimensioni:** L'immagine verrà ridimensionata a 28x28 pixel.
- **Colore:** Preferibilmente uno sfondo nero con il numero bianco. Se hai uno sfondo chiaro, lo script inverte automaticamente i colori.

#### Esecuzione dello Script

1. **Assicurati di essere nell'ambiente virtuale e di avere il modello addestrato (`mnist_model.h5`).**

2. **Esegui lo script passando il percorso dell'immagine come argomento:**

   ```bash
   python3 predict_number.py percorso_immagine.png
   ```

   Ad esempio, se hai un'immagine chiamata `cifra.png` nella stessa directory:

   ```bash
   python3 predict_number.py cifra.png
   ```

#### Output Atteso

Lo script caricherà il modello e prevede il numero nell'immagine, mostrando anche la confidenza della previsione.

Esempio di output:

```
Modello caricato con successo.
1/1 [==============================] - 0s 32ms/step
Il numero previsto è: 6 con una confidenza di 28.39%
```

> **Nota:** Una confidenza bassa (ad esempio, 28.39%) indica che il modello non è sicuro della sua previsione. Ciò potrebbe essere dovuto a problemi di preprocessing, qualità dell'immagine o performance del modello.

## Dettagli del Funzionamento

### `cifra.py`

1. **Caricamento del Dataset MNIST:**

   ```python
   (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
   ```

   - **x_train, x_test:** Immagini dei numeri (28x28 pixel).
   - **y_train, y_test:** Etichette corrispondenti (0-9).

2. **Normalizzazione dei Dati:**

   ```python
   x_train = x_train.astype('float32') / 255.0
   x_test = x_test.astype('float32') / 255.0
   ```

   Converte i valori dei pixel da 0-255 a 0-1.

3. **Aggiunta della Dimensione del Canale:**

   ```python
   x_train = np.expand_dims(x_train, -1)
   x_test = np.expand_dims(x_test, -1)
   ```

   Aggiunge una dimensione per il canale (grayscale).

4. **Costruzione del Modello:**

   ```python
   model = keras.Sequential([
       layers.Flatten(input_shape=(28, 28, 1)),
       layers.Dense(128, activation='relu'),
       layers.Dense(10, activation='softmax')
   ])
   ```

   - **Flatten:** Converte l'immagine 2D in un vettore 1D.
   - **Dense 1:** Strato nascosto con 128 neuroni e attivazione ReLU.
   - **Dense 2:** Strato di output con 10 neuroni (una per ogni classe) e softmax.

5. **Compilazione del Modello:**

   ```python
   model.compile(
       optimizer='adam',
       loss='sparse_categorical_crossentropy',
       metrics=['accuracy']
   )
   ```

   Specifica l'ottimizzatore, la funzione di perdita e le metriche.

6. **Addestramento del Modello:**

   ```python
   history = model.fit(
       x_train, y_train,
       epochs=5,
       batch_size=32,
       validation_split=0.1
   )
   ```

   Addestra il modello per 5 epoche con batch di 32 campioni e utilizza il 10% dei dati per la validazione.

7. **Valutazione del Modello:**

   ```python
   test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
   print(f"\nTest accuracy: {test_accuracy:.4f}")
   ```

   Valuta le prestazioni del modello sul set di test.

8. **Salvataggio del Modello:**

   ```python
   model_save_path = 'mnist_model.h5'
   model.save(model_save_path)
   print(f"Modello salvato in: {os.path.abspath(model_save_path)}")
   ```

   Salva il modello addestrato in un file `.h5`.

9. **Caricamento del Modello Salvato e Previsioni:**

   ```python
   loaded_model = keras.models.load_model(model_save_path)
   print("Modello caricato con successo.")
   
   predictions = loaded_model.predict(x_test)
   
   num_predictions = 5
   for i in range(num_predictions):
       print(f"Predizione per l'immagine {i}: {np.argmax(predictions[i])}, Etichetta reale: {y_test[i]}")
   ```

   Carica il modello salvato e mostra le prime 5 previsioni sul set di test.

### `predict_number.py`

1. **Import delle Librerie Necessarie:**

   ```python
   import tensorflow as tf
   from tensorflow import keras
   import numpy as np
   from PIL import Image, ImageOps
   import sys
   import os
   import matplotlib.pyplot as plt
   ```

2. **Funzione di Preprocessing dell'Immagine:**

   ```python
   def load_and_preprocess_image(image_path):
       try:
           img = Image.open(image_path).convert('L')  # Converti in scala di grigi
       except Exception as e:
           print(f"Errore nell'aprire l'immagine: {e}")
           sys.exit(1)
       
       img = img.resize((28, 28), Image.LANCZOS)  # Ridimensiona a 28x28
       
       img_array = np.array(img)
       if np.mean(img_array) > 127:
           img = ImageOps.invert(img)
           img_array = np.array(img)
       
       img_array = img_array.astype('float32') / 255.0
       
       img_array = np.expand_dims(img_array, axis=-1)  # Shape: (28, 28, 1)
       img_array = np.expand_dims(img_array, axis=0)   # Shape: (1, 28, 28, 1)
       
       return img_array
   ```

   - **Conversione in scala di grigi**
   - **Ridimensionamento a 28x28 pixel**
   - **Inversione dei colori se necessario**
   - **Normalizzazione dei pixel**
   - **Aggiunta delle dimensioni per il canale e il batch**

3. **Funzione Principale: Caricamento del Modello e Previsione**

   ```python
   def main():
       if len(sys.argv) != 2:
           print("Uso corretto: python predict_number.py <percorso_immagine>")
           sys.exit(1)
       
       image_path = sys.argv[1]
       
       if not os.path.exists(image_path):
           print(f"Errore: Il file '{image_path}' non esiste.")
           sys.exit(1)
       
       img = load_and_preprocess_image(image_path)
       
       model_path = 'mnist_model.h5'
       if not os.path.exists(model_path):
           print(f"Errore: Il modello '{model_path}' non è stato trovato. Assicurati di aver eseguito 'cifra.py' prima.")
           sys.exit(1)
       
       model = keras.models.load_model(model_path)
       print("Modello caricato con successo.")
       
       predictions = model.predict(img)
       predicted_digit = np.argmax(predictions[0])
       confidence = np.max(predictions[0])
       
       print(f"Il numero previsto è: {predicted_digit} con una confidenza di {confidence*100:.2f}%")
   ```

   - **Verifica degli argomenti della riga di comando**
   - **Preprocessing dell'immagine**
   - **Caricamento del modello salvato**
   - **Effettuare la previsione e mostrare il risultato**

4. **Esecuzione del Main:**

   ```python
   if __name__ == "__main__":
       main()
   ```

## Risoluzione dei Problemi

Durante l'esecuzione degli script, potresti incontrare alcuni errori comuni. Ecco come risolverli:

### 1. `ImportError: cannot import name 'Resampling' from 'PIL'`

**Causa:** La tua versione di Pillow è inferiore a 10.0.0, che non supporta `Resampling`.

**Soluzione:**

- **Aggiorna Pillow:**

  ```bash
  pip install --upgrade pillow
  ```

- **Oppure usa `Image.LANCZOS` direttamente:**

  Se non puoi aggiornare Pillow, modifica `predict_number.py` sostituendo `Resampling.LANCZOS` con `Image.LANCZOS`:

  ```python
  img = img.resize((28, 28), Image.LANCZOS)
  ```

### 2. Avvisi di TensorFlow su cuDNN, cuBLAS e TensorRT

**Causa:** TensorFlow sta cercando di registrare plugin GPU come cuDNN e cuBLAS, ma sembra che ci siano conflitti o mancanza di alcune librerie GPU.

**Soluzione:**

- **Ignora gli Avvisi se non usi GPU:** Se stai eseguendo il modello su CPU, questi avvisi possono essere ignorati.

- **Nascondi gli Avvisi di TensorFlow:** Aggiungi le seguenti righe all'inizio di `predict_number.py` e `cifra.py` per ridurre i messaggi di log:

  ```python
  import os
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = DEBUG, 1 = INFO, 2 = WARNING, 3 = ERROR
  ```

- **Configura correttamente l'ambiente GPU:** Se desideri utilizzare una GPU, assicurati di aver installato correttamente CUDA, cuDNN e TensorRT seguendo la [Guida Ufficiale di TensorFlow per l'Installazione con GPU](https://www.tensorflow.org/install/gpu).

### 3. Bassa Confidenza nelle Previsioni

**Causa:** Diverse possibili ragioni, tra cui preprocessing inadeguato, qualità dell'immagine, modello non addestrato correttamente.

**Soluzione:**

- **Verifica il Preprocessing:** Assicurati che l'immagine sia preprocessata correttamente. Aggiungi una visualizzazione dell'immagine preprocessata:

  ```python
  import matplotlib.pyplot as plt

  def load_and_preprocess_image(image_path):
      # ... [il tuo codice di preprocessing] ...

      # Visualizza l'immagine preprocessata
      plt.imshow(img_array[0, :, :, 0], cmap='gray')
      plt.title("Immagine Preprocessata")
      plt.show()

      return img_array
  ```

- **Migliora l'Addestramento del Modello:** Aumenta il numero di epoche, aggiungi strati o neuroni, utilizza tecniche di regolarizzazione come il dropout.

- **Assicurati che il Modello sia Addestrato Correttamente:** Verifica che il modello abbia un'alta accuratezza sul set di test.

- **Usa una Rete Neurale Convoluzionale (CNN):** Le CNN sono più efficaci per il riconoscimento delle immagini rispetto ai semplici modelli completamente connessi.

## Ulteriori Miglioramenti

### 1. Aggiungi Data Augmentation

Aumenta la varietà del tuo dataset utilizzando tecniche di data augmentation per migliorare la generalizzazione del modello.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Addestra il modello utilizzando data augmentation
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(x_test, y_test)
)
```

### 2. Utilizza una Rete Neurale Convoluzionale (CNN)

Le CNN sono più adatte per il riconoscimento delle immagini.

```python
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

### 3. Visualizza le Previsioni

Aggiungi una funzione per visualizzare l'immagine e la previsione del modello.

```python
def display_predictions(predictions, image_path):
    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert('L').resize((28, 28), Image.LANCZOS)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f"Predizione: {np.argmax(predictions)}, Confidenza: {np.max(predictions)*100:.2f}%")
    plt.show()
```

Chiamala nel `main` dopo la previsione:

```python
display_predictions(predictions[0], image_path)
```

## Risorse Utili

- [Documentazione di TensorFlow](https://www.tensorflow.org/api_docs)
- [Documentazione di Keras](https://keras.io/api/)
- [Guida Ufficiale di TensorFlow per l'Installazione con GPU](https://www.tensorflow.org/install/gpu)
- [Documentazione di Pillow](https://pillow.readthedocs.io/en/stable/)
- [Dataset MNIST](http://yann.lecun.com/exdb/mnist/)

## Conclusione

Questo progetto fornisce una base solida per costruire e utilizzare un sistema di riconoscimento dei numeri scritti a mano. Seguendo questo README, dovresti essere in grado di configurare l'ambiente, addestrare il modello e utilizzare il sistema di previsione. Per migliorare ulteriormente le prestazioni, considera di esplorare architetture più complesse, tecniche di data augmentation e altre strategie di ottimizzazione.

Se incontri ulteriori problemi o hai domande, non esitare a chiedere! info@soluzioniweb.net
