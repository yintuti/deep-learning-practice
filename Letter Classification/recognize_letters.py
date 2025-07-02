from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
import sys

character_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def format_image(file_path):
    
    image = Image.open(file_path).convert('L')
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    
    pixels = np.array(image) / 255.0

    pixels = pixels.T
    
    ready_image = pixels.reshape(1, 28, 28, 1)
    
    return ready_image


print(">>> Loading the letters model 'letters_model.h5'...")
try:
    neural_network = keras.models.load_model('letters_model.h5')
except Exception as e:
    print(f"!!! ERROR: Could not load the file 'letters_model.h5'.")
    print(f"!!! Error: {e}")
    sys.exit()

if len(sys.argv) < 2:
    print("\nUsage: python recognize_letters.py <path_to_your_image.png>")
    sys.exit()

test_file = sys.argv[1]
print(f">>> Processing the image '{test_file}'...")

try:
    formatted_image = format_image(test_file)
except FileNotFoundError:
    print(f"!!! ERROR: The file '{test_file}' was not found.")
    sys.exit()

raw_prediction = neural_network.predict(formatted_image)
predicted_index = np.argmax(raw_prediction)

recognized_character = character_map[predicted_index]

print("\n======================================")
print(f"  The character in the image is: {recognized_character}")
print("======================================\n")