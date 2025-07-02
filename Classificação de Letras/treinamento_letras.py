import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

dataset, info = tfds.load('emnist/byclass', split='train', with_info=True, as_supervised=True)

NUM_CLASSES = info.features['label'].num_classes
print(f"Dataset EMNIST carregado. Número de classes: {NUM_CLASSES}")

def normalizar_e_formatar(imagem, rotulo):
    imagem = tf.cast(imagem, tf.float32) / 255.0
    
    rotulo = tf.one_hot(rotulo, depth=NUM_CLASSES)
    return imagem, rotulo

TAMANHO_LOTE = 128
dataset_preparado = dataset.map(normalizar_e_formatar).shuffle(buffer_size=10000).batch(TAMANHO_LOTE)


meu_modelo = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

meu_modelo.summary()

meu_modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nIniciando o treinamento de letras...")
print("-" * 50)

meu_modelo.fit(
    dataset_preparado,
    epochs=5
)

print("-" * 50)
print("Treinamento finalizado")

meu_modelo.save('modelo_letras.h5')
print(f"\nModelo de letras salvo com sucesso em 'modelo_letras.h5'")