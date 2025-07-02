import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
import sys

def prepare_image(image_path):
    img = Image.open(image_path).convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img_array = np.array(img, dtype='float32') / 255.0
    img_array = img_array.reshape(1, 28, 28)
    return img_array

try:
    model = keras.models.load_model('numbers_model.h5')
    print("Model 'numbers_model.h5' loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    sys.exit()

if len(sys.argv) < 2:
    print("\nUsage: python recognize_numbers.py <path_to_your_image.png>")
    sys.exit()

test_image_path = sys.argv[1]

try:
    processed_image = prepare_image(test_image_path)
    print(f"Image '{test_image_path}' processed.")
except FileNotFoundError:
    print(f"Error: The file '{test_image_path}' was not found.")
    sys.exit()

print("Making prediction...")
predictions = model.predict(processed_image)
predicted_digit = np.argmax(predictions)

print("\n-------------------------------")
print(f"  The digit in the image is: {predicted_digit}")
print("-------------------------------")