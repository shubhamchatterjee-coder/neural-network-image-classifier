from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow import keras
import base64
from PIL import Image
import io
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ─── Load Both Models ───────────────────────────────────────
print("Loading models... please wait...")

# MNIST Model
mnist_model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Train MNIST model quickly
from tensorflow.keras.datasets import mnist
(x_train, y_train), _ = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train_flat = x_train.reshape(-1, 784)
mnist_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
print("Training MNIST model (quick)...")
mnist_model.fit(x_train_flat, y_train, epochs=5,
                batch_size=256, verbose=0)
print("MNIST model ready!")

# CIFAR-10 Model
from tensorflow.keras.datasets import cifar10
CIFAR_CLASSES = ['Airplane','Car','Bird','Cat','Deer',
                 'Dog','Frog','Horse','Ship','Truck']

cifar_model = keras.Sequential([
    keras.layers.Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(32,32,3)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32,(3,3),padding='same',activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.3),
    keras.layers.Conv2D(128,(3,3),padding='same',activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.4),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

(x_c_train, y_c_train), _ = cifar10.load_data()
x_c_train = x_c_train.astype('float32') / 255.0
y_c_train = y_c_train.flatten()
cifar_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
print("Training CIFAR-10 model (quick)...")
cifar_model.fit(x_c_train, y_c_train, epochs=10,
                batch_size=128, verbose=0)
print("CIFAR-10 model ready!")
print("Both models loaded! Visit http://127.0.0.1:5000")

# ─── Routes ─────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data      = request.get_json()
    model_type = data['model']
    img_data  = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    img       = Image.open(io.BytesIO(img_bytes))

    if model_type == 'mnist':
        img = img.convert('L').resize((28, 28))
        arr = np.array(img).astype('float32') / 255.0
        arr = arr.reshape(1, 784)
        pred = mnist_model.predict(arr, verbose=0)[0]
        label     = str(np.argmax(pred))
        confidence = float(np.max(pred)) * 100
        all_probs = {str(i): round(float(pred[i])*100, 1)
                     for i in range(10)}
        return jsonify({'label': label,
                        'confidence': round(confidence, 1),
                        'all_probs': all_probs})
    else:
        img = img.convert('RGB').resize((32, 32))
        arr = np.array(img).astype('float32') / 255.0
        arr = arr.reshape(1, 32, 32, 3)
        pred = cifar_model.predict(arr, verbose=0)[0]
        idx       = int(np.argmax(pred))
        label     = CIFAR_CLASSES[idx]
        confidence = float(np.max(pred)) * 100
        all_probs = {CIFAR_CLASSES[i]: round(float(pred[i])*100, 1)
                     for i in range(10)}
        return jsonify({'label': label,
                        'confidence': round(confidence, 1),
                        'all_probs': all_probs})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
