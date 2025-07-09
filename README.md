# Deep Learning Projects

This repository contains two distinct deep learning projects for image recognition:

## Project Overview

### 1. [Number Recognition](./Number%20Recognition/)
A digit recognition system based on the MNIST dataset using neural networks.

**Features:**
- Recognizes handwritten digits (0-9)
- Uses the MNIST dataset for training
- Simple neural network architecture
- Fast and accurate digit recognition

### 2. [Letter Classifier](./Letter%20Classifier/)
A character recognition system that can identify letters and numbers using the EMNIST dataset.

**Features:**
- Recognizes uppercase and lowercase letters (A-Z, a-z)
- Recognizes digits (0-9)
- Uses the EMNIST dataset for training
- Comprehensive character recognition

## Quick Start

Each project is independent and can be run separately. Navigate to the desired project folder and follow its specific README for detailed instructions.

## Requirements

All projects require:
- Python 3.7+
- TensorFlow
- NumPy
- PIL (Pillow)

Install dependencies:
```bash
pip install tensorflow numpy pillow
```

## Project Structure

```
Deep Learning/
├── README.md                    # This file
├── Number Recognition/          # Digit recognition
│   ├── README.md
│   ├── train_numbers.py
│   ├── recognize_numbers.py
│   └── numbers_model.h5
└── Letter Classifier/           # Character recognition
    ├── README.md
    ├── train_letters.py
    ├── recognize_letters.py
    └── letters_model.h5
```