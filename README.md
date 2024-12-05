
# Skin Disease Classification Using Deep Learning with PyTorch

This project implements a deep learning model to classify various skin diseases using convolutional neural networks (CNNs) in PyTorch. The model is trained to recognize different skin disease categories based on images. The dataset is preprocessed and augmented for enhanced model performance.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Environment Setup](#environment-setup)
- [Code Overview](#code-overview)
- [Model Training](#model-training)
- [Results](#results)
- [Using Pre-trained ResNet50](#using-pre-trained-resnet50)
- [License](#license)

## Project Overview

This project uses a custom CNN and pre-trained ResNet50 model to classify images of skin diseases into 10 categories. The project follows the steps of data preprocessing, image augmentation, model training, and evaluation to ensure optimal performance on unseen data.

### Skin Disease Categories:
1. Eczema
2. Melanoma
3. Atopic Dermatitis
4. Basal Cell Carcinoma
5. Melanocytic Nevi
6. Benign Keratosis-like Lesions
7. Psoriasis
8. Seborrheic Keratoses
9. Tinea
10. Warts

## Dataset

The dataset contains images of skin diseases categorized by their respective labels. The dataset is organized into folders, with each folder named after the skin disease and containing images corresponding to that disease.

Source: https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset
- **Root Directory:** Images are stored in a directory structure where each folder corresponds to a different skin disease.
- **Data Augmentation:** Images are augmented using techniques such as rotation, flipping, and cropping to increase the diversity of the training data.

### Dataset Preprocessing:
- Images are resized to a target size of 180x180 pixels.
- Gaussian blur, denoising, and sharpening filters are applied to the images to enhance features.
- Augmentation functions such as horizontal flip, vertical flip, rotation, and cropping are applied.

## Environment Setup

### Prerequisites:
- Python 3.7+
- PyTorch
- OpenCV
- Pillow
- scikit-learn
- Pandas
- NumPy

### Installation:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/skin-disease-classification.git
   cd skin-disease-classification
   ```

2. Create a virtual environment and install the dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Install any additional dependencies if needed:
   ```bash
   pip install torch torchvision opencv-python pillow scikit-learn numpy
   ```

## Code Overview

### 1. Data Preprocessing and Augmentation
The `create_data()` function processes the dataset by organizing image paths and labels into a DataFrame. The `augment_dataset()` function applies various augmentations to increase the training data.

### 2. Image Preprocessing Functions
Several functions are defined to preprocess the images, including resizing, filtering (Gaussian blur, sharpening, denoising), and augmentations (flip, rotation, cropping).

### 3. CNN Model Architecture
A custom CNN is defined in the `SkinDiseaseClassifier` class. The network uses two convolutional layers followed by fully connected layers for classification. Dropout is applied to prevent overfitting.

### 4. Model Training
The model is trained using Cross-Entropy Loss and Adam optimizer. The training loop consists of training and validation phases with periodic accuracy reporting. Learning rate scheduling is also included.

### 5. Evaluation
The model's performance is evaluated using a test set after training, reporting loss and accuracy.

## Model Training

### 1. Dataset Loading:
- The dataset is split into training, validation, and test sets using the `train_test_split()` function from `scikit-learn`.

### 2. Training Loop:
- The model is trained for 3 epochs (can be adjusted in the code) with a batch size of 64.
- During each epoch, the training and validation loss and accuracy are computed.

### 3. Evaluation:
- The model is evaluated on the test set after training to assess its final performance.

### Hyperparameters:
- Batch Size: 64
- Learning Rate: 0.001
- Epochs: 3

## Results

Once the training is complete, the model will output the following metrics for each epoch:
- Training loss and accuracy
- Validation loss and accuracy

Test results will be provided at the end of the training process.

## Using Pre-trained ResNet50

In addition to training a custom CNN model, this project also utilizes the pre-trained ResNet50 model from `torchvision.models`. The last fully connected layer is replaced to fit the 10-class skin disease classification task.

- The ResNet50 model is fine-tuned by freezing the earlier layers and training only the final fully connected layer.
- This approach helps leverage the pre-trained features of ResNet50, which can improve model performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to clone the repo, experiment with the code, and improve the model's performance!
