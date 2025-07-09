# Letter Classifier

This project implements a comprehensive character recognition system that can identify uppercase letters, lowercase letters, and digits using the EMNIST dataset and neural networks with TensorFlow.

## Project Structure

```
Letter Classifier/
├── train_letters.py          # Training script
├── recognize_letters.py      # Recognition script
├── letters_model.h5         # Trained model (will be created)
├── letterALower.png         # Example test image
├── letterBUpper.png         # Example test image
└── letterGUpper.png         # Example test image
```

## How to Use

### 1. Train the Model

Run the training script to create the model:

```bash
py train_letters.py
```

This will:
- Load the EMNIST dataset (Extended MNIST)
- Train a neural network for 5 epochs
- Save the model as `letters_model.h5`

### 2. Recognize Characters

Use the recognition script to identify characters in images:

```bash
py recognize_letters.py path/to/your/image.png
```

## Model Architecture

The neural network consists of:
- Input layer: Flattened 28x28 pixel images
- Hidden layer: 512 neurons with ReLU activation
- Dropout layer: 20% dropout for regularization
- Output layer: 62 neurons (one for each character) with softmax activation

## Supported Characters

The model can recognize:
- **Digits**: 0-9 (10 characters)
- **Uppercase letters**: A-Z (26 characters)
- **Lowercase letters**: a-z (26 characters)
- **Total**: 62 different characters

## Dataset

This project uses the EMNIST (Extended MNIST) dataset, which contains:
- Handwritten characters from the NIST Special Database 19
- 28x28 pixel grayscale images
- Balanced dataset with multiple writers
- High-quality character samples

## Usage Example

```bash
py train_letters.py

py recognize_letters.py letterA.png
```

## Requirements

- Python 3.7+
- TensorFlow
- TensorFlow Datasets
- NumPy
- PIL (Pillow)

Install dependencies:

```bash
pip install tensorflow tensorflow-datasets numpy pillow
```

## Image Requirements

For best recognition results:
- Use grayscale images
- Ensure the character is centered
- Use white characters on black background (or vice versa)
- Image will be automatically resized to 28x28 pixels
- The image will be inverted automatically if needed