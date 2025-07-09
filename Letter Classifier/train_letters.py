import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

dataset, info = tfds.load('emnist/byclass', split='train', with_info=True, as_supervised=True)

NUM_CLASSES = info.features['label'].num_classes
print(f"EMNIST dataset loaded. Number of classes: {NUM_CLASSES}")

def normalize_and_format(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, depth=NUM_CLASSES)
    return image, label

BATCH_SIZE = 128
prepared_dataset = dataset.map(normalize_and_format).shuffle(buffer_size=10000).batch(BATCH_SIZE)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nStarting letter model training...")
print("-" * 50)

model.fit(
    prepared_dataset,
    epochs=5
)

print("-" * 50)
print("Training finished")

model.save('letters_model.h5')
print(f"\nLetters model successfully saved as 'letters_model.h5'")