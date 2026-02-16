# 🐶 Dog Breed Recognition: From Hand-Crafted Features to Transfer Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📖 Project Overview
This project implements an AI agent capable of performing **Fine-Grained Image Classification** on 50 different dog breeds. 

Unlike general object detection, dog breed recognition is challenging due to **high inter-class similarity** (e.g., distinguishing *Miniature Poodle* from *Toy Poodle*) and **high intra-class variation**. 

We adopted a **comparative approach**, implementing and evaluating three distinct methodologies to demonstrate the evolution of computer vision techniques:
1.  **MLP (Multi-Layer Perceptron):** Baseline using hand-crafted feature engineering.
2.  **Simple CNN:** Deep learning model trained from scratch (to demonstrate data scarcity challenges).
3.  **ResNet-50 (Transfer Learning):** State-of-the-art approach utilizing pre-trained ImageNet weights.

## 📊 Dataset
* **Source:** [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) (Subset of ImageNet).
* **Classes:** 50 distinct dog breeds.
* **Size:** Approximately 8,000 images.
* **Split:** 80% Training, 10% Validation, 10% Testing.
* **Preprocessing:**
    * MLP: Resized to 128x128, Feature Extraction.
    * ResNet: Resized to 256x256, Center Cropped to 224x224, Normalized.

## 🛠️ Methodologies & Models

### 1. MLP with Hand-Crafted Features (Baseline)
* **File:** `MLP_CLASSIFICATION.ipynb`
* **Approach:** Traditional Machine Learning pipeline.
* **Feature Engineering:** We implemented a custom `OptimizedFeatureExtractor` to convert images into **79-dimensional vectors**. Features include:
    * HSV Color Histograms.
    * Color Channel Statistics (Mean, Std).
    * Texture Features (Sobel Gradients).
    * Edge Features (Canny Edge Detection).
* **Architecture:** 3 Hidden Layers (512 $\to$ 256 $\to$ 128) with ReLU activation.
* **Result:** Surprisingly robust (**81.63% accuracy**), proving the effectiveness of manual feature engineering.

### 2. Simple CNN (Trained from Scratch)
* **File:** `CNN_v0.1_1000_Epochs.ipynb`
* **Approach:** Training a Convolutional Neural Network from random initialization.
* **Architecture:** 3 Convolutional Blocks (Conv2D -> BatchNorm -> MaxPool).
* **Training:** Trained for **1,000 Epochs**.
* **Result:** **Severe Overfitting**. While training accuracy reached >90%, test accuracy stalled at **~26.9%**. This highlights the difficulty of learning visual filters from scratch with limited data.

### 3. ResNet-50 (Transfer Learning)
* **File:** `CNN_ResNet50_Transfer_Learning.ipynb`
* **Approach:** Leveraging a model pre-trained on ImageNet.
* **Technique:** * Frozen Backbone (Feature Extractor).
    * Modified Fully Connected Head ($2048 \to 50$ classes).
* **Training:** Fine-tuned for only **10 Epochs**.
* **Result:** **Best Performance (87.48% accuracy)**. The model effectively bridged the semantic gap by reusing learned visual hierarchies.

## 🏆 Performance Comparison

| Model | Method | Training Accuracy | Test Accuracy | Training Time | Key Observation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP** | Hand-Crafted Features | 98.27% | **81.63%** | ~2 mins | Strong baseline, lacks spatial invariance. |
| **Simple CNN** | Trained from Scratch | >90.00% | **26.93%** | ~8 hours | Catastrophic overfitting due to data scarcity. |
| **ResNet-50** | Transfer Learning | 98.43% | **87.48%** | ~45 mins | **Optimal solution**; excellent generalization. |

## 🖼️ Visualization Results

*(Place your confusion matrix or prediction sample images here if available)*

* **Confusion Matrix (ResNet-50):** Shows high diagonal density, indicating correct classification for most breeds.
* **Failure Cases:** Confusion primarily occurs between size-variant breeds (e.g., *Toy Poodle* vs. *Miniature Poodle*) due to loss of scale information during resizing.

## 🚀 Installation & Usage

The code is designed to run in **Google Colab**.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Dog-Breed-Recognition.git](https://github.com/YourUsername/Dog-Breed-Recognition.git)
    ```
2.  **Upload Data:**
    * Download the dataset and upload it to your Google Drive.
    * Update the `DATASET_PATH` variable in the notebooks to point to your Drive folder.
3.  **Install Dependencies:**
    ```python
    pip install torch torchvision tensorflow scikit-learn opencv-python matplotlib seaborn pandas tqdm
    ```
4.  **Run the Notebooks:**
    * Open any `.ipynb` file in Google Colab or Jupyter Notebook and execute the cells.

## 👥 Contributors
* **[Your Name]**: Model implementation (CNN/ResNet), Report writing.
* **[Team Member Name]**: MLP implementation, Feature extraction pipeline.
* **[Team Member Name]**: KNN Baseline, Data augmentation.

## 📜 License
This project is for academic purposes (CSCI218 Group Project).

## 🙏 Acknowledgments
* Course: CSCI218 Foundations of AI (University of Wollongong).
* Dataset provided by [Stanford Vision Lab](http://vision.stanford.edu/).
