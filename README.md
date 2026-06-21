# Nature Image Classification with CNN

A PyTorch-based convolutional neural network (CNN) for classifying selected nature-related classes from the CIFAR-100 dataset. The project includes the full training notebook and a Flask web app where users can upload an image and view the model's prediction with confidence scores.

## Project Overview

This project was built as part of my deep learning practice after completing a CNN-focused course. The goal was to train a simple CNN model, evaluate its performance, and turn it into a small web application that others can test.

The model classifies images into 15 selected CIFAR-100 classes:

- orchid
- poppy
- rose
- sunflower
- tulip
- fox
- porcupine
- possum
- raccoon
- skunk
- bee
- beetle
- butterfly
- caterpillar
- cockroach

## Live Demo

Try the deployed Flask app here:

[Nature Classification CNN](https://nature-classification-cnn.onrender.com/)

## Features

- CNN model built with PyTorch
- CIFAR-100 subset classification
- Data preprocessing and augmentation
- Training and validation workflow in Jupyter Notebook
- Flask web interface for image upload and prediction
- Top prediction with confidence score
- Top 3 class probabilities
- CPU-compatible inference

## Project Structure

```text
Nature_classification/
├── app.py
├── model.py
├── models/
│   └── nature_cnn.pth
├── notebooks/
│   └── nature-cifar100-cnn-training.ipynb
├── static/
│   └── css/
│       └── style.css
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
```

## Model Architecture

The model is a simple CNN with:

- 3 convolutional layers
- ReLU activation functions
- Max pooling layers
- Flatten layer
- Fully connected layer
- Dropout for regularization
- Final classification layer for 15 classes

## Data Preprocessing

The training pipeline uses:

- Random horizontal flipping
- Random rotation
- Tensor conversion
- CIFAR-100 mean and standard deviation normalization

The validation and prediction pipelines use:

- Image resizing to 32x32
- Tensor conversion
- CIFAR-100 normalization

## Result

The model was trained for 25 epochs on CPU.

Final validation result:

```text
Validation Accuracy: 66.00%
```

This result is reasonable for a simple CNN trained on a subset of CIFAR-100. The goal of this project is not only high accuracy, but also understanding the full workflow from training to deployment.

## How to Run Locally

Clone the repository:

```bash
git clone https://github.com/swayamkr03/Nature_classification_CNN.git
cd Nature_classification_CNN
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask app:

```bash
python app.py
```

Open in your browser:

```text
http://127.0.0.1:5000
```

## Training Notebook

The complete training workflow is available in:

```text
notebooks/nature-cifar100-cnn-training.ipynb
```

The notebook includes:

- Dataset loading
- Class filtering
- Data transformation
- CNN model definition
- Training loop
- Validation metrics
- Prediction visualization
- Model saving

## Important Note

This model was trained on CIFAR-100 images, which are small 32x32 images. Because of this, predictions on real-world high-resolution images may not always be accurate. The uploaded image is resized to 32x32 before prediction to match the training setup.

## Future Improvements

Possible improvements for this project include:

- Improve validation accuracy with a deeper CNN architecture
- Tune hyperparameters such as learning rate, batch size, dropout rate, and number of epochs
- Add learning rate scheduling
- Use batch normalization after convolutional layers
- Try transfer learning with pretrained models such as ResNet, EfficientNet, or MobileNet
- Train on a larger and more realistic nature image dataset
- Add more classes beyond the current CIFAR-100 subset
- Use stronger data augmentation techniques
- Add confusion matrix and per-class accuracy analysis
- Save and display training curves in the README
- Add model comparison between custom CNN and pretrained models
- Improve the Flask UI with image preview and better error handling
- Deploy the app online using Render, Railway, Hugging Face Spaces, or another hosting platform

## Tech Stack

- Python
- PyTorch
- Torchvision
- Flask
- HTML
- CSS
- Jupyter Notebook

## What I Learned

Through this project, I practiced:

- Building CNNs with PyTorch
- Preparing image datasets
- Applying image transformations and normalization
- Training and evaluating classification models
- Saving and loading model weights
- Connecting a trained ML model to a Flask web application
- Structuring a machine learning project for GitHub
