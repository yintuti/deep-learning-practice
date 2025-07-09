# Number Recognition

This project implements a handwritten digit recognition system using the MNIST dataset and neural networks with TensorFlow.

## Project Structure

```
Number Recognition/
├── train_numbers.py          # Training script
├── recognize_numbers.py      # Recognition script
├── numbers_model.h5         # Trained model (will be created)
└── number4.png              # Example test image
```

## How to Use

### 1. Train the Model

Run the training script to create the model:

```bash
py train_numbers.py
```

This will:
- Load the MNIST dataset
- Train a neural network for 10 epochs
- Save the model as `numbers_model.h5`

### 2. Recognize Digits

Use the recognition script to identify digits in images:

```bash
py recognize_numbers.py path/to/your/image.png
```

## Model Architecture

The neural network consists of:
- Input layer: Flattened 28x28 pixel images
- Hidden layer: 512 neurons with ReLU activation
- Dropout layer: 20% dropout for regularization
- Output layer: 10 neurons (one for each digit 0-9) with softmax activation

## Dataset

This project uses the MNIST dataset, which contains:
- 60,000 training images
- 10,000 test images
- 28x28 pixel grayscale images
- Handwritten digits from 0 to 9

## Requirements

- Python 3.7+
- TensorFlow
- NumPy
- PIL (Pillow)

Install dependencies:

```bash
pip install tensorflow numpy pillow
```

## Image Requirements

For best recognition results:
- Use grayscale images
- Ensure the digit is centered
- Use white digits on black background (or vice versa)
- Image will be automatically resized to 28x28 pixels