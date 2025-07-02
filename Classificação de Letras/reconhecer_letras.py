from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
import sys

mapa_caracteres = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def formata_imagem(caminho_do_arquivo):
    """Prepara uma imagem para ser analisada pelo modelo de letras EMNIST."""
    
    imagem = Image.open(caminho_do_arquivo).convert('L')
    imagem = ImageOps.invert(imagem)
    imagem = imagem.resize((28, 28))
    
    pixels = np.array(imagem) / 255.0

    pixels = pixels.T
    
    imagem_pronta = pixels.reshape(1, 28, 28, 1)
    
    return imagem_pronta


print(">>> Carregando o modelo de letras 'modelo_letras.h5'...")
try:
    rede_neural = keras.models.load_model('modelo_letras.h5')
except Exception as e:
    print(f"!!! FALHA: Não consegui carregar o arquivo 'modelo_letras.h5'.")
    print(f"!!! Erro: {e}")
    sys.exit()

if len(sys.argv) < 2:
    print("\nUso: py reconhecer_letras.py <caminho_para_sua_imagem.png>")
    sys.exit()

arquivo_teste = sys.argv[1]
print(f">>> Processando a imagem '{arquivo_teste}'...")

try:
    imagem_formatada = formata_imagem(arquivo_teste)
except FileNotFoundError:
    print(f"!!! ERRO: O arquivo '{arquivo_teste}' não foi encontrado.")
    sys.exit()

predicao_bruta = rede_neural.predict(imagem_formatada)
indice_predito = np.argmax(predicao_bruta)

caractere_reconhecido = mapa_caracteres[indice_predito]

print("\n======================================")
print(f"  O caractere na imagem é: {caractere_reconhecido}")
print("======================================\n")