import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np

print("Loading the MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train_categorical = to_categorical(y_train, num_classes=10)
y_test_categorical = to_categorical(y_test, num_classes=10)
print("Data prepared successfully.")

print("Building the Keras model...")
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print("Model compiled")

print("\nStarting model training...")
history = model.fit(x_train, y_train_categorical,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.1)
print("Training finished")

score = model.evaluate(x_test, y_test_categorical, verbose=0)
print(f'\nFinal accuracy on test set: {score[1]:.4f}')

model.save('numbers_model.h5')
print(f"\nModel successfully saved as 'numbers_model.h5'")