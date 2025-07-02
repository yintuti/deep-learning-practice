import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
import sys

def preparar_imagem(caminho_imagem):
    img = Image.open(caminho_imagem).convert('L')
    
    img = ImageOps.invert(img)
    
    img = img.resize((28, 28))
    
    img_array = np.array(img, dtype='float32') / 255.0
    
    img_array = img_array.reshape(1, 28, 28)
    
    return img_array

try:
    modelo = keras.models.load_model('modelo.h5')
    print("Modelo 'modelo.h5' carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    sys.exit()

if len(sys.argv) < 2:
    print("\nUso: py reconhecer.py <caminho_para_sua_imagem.png>")
    sys.exit()

caminho_imagem_teste = sys.argv[1]

try:
    imagem_processada = preparar_imagem(caminho_imagem_teste)
    print(f"Imagem '{caminho_imagem_teste}' processada.")
except FileNotFoundError:
    print(f"Erro: O arquivo '{caminho_imagem_teste}' não foi encontrado.")
    sys.exit()

print("Fazendo a previsão...")
previsoes = modelo.predict(imagem_processada)
digito_previsto = np.argmax(previsoes)

print("\n-------------------------------")
print(f"  O dígito na imagem é: {digito_previsto}")
print("-------------------------------")