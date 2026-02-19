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
1. **MLP (Multi-Layer Perceptron):** Baseline using hand-crafted feature engineering.
2. **Simple CNN:** Deep learning model trained from scratch (to demonstrate data scarcity challenges).
3. **ResNet-50 (Transfer Learning):** State-of-the-art approach utilizing pre-trained ImageNet weights.

## 📊 Dataset
* **Source:** [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) (Subset of ImageNet).
* **Classes:** 50 distinct dog breeds.
* **Size:** Approximately 8,000 images.

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
* **File:** `AI Project — CNN v0.1 (1000 Epochs).ipynb`
* **Approach:** Training a Convolutional Neural Network from random initialization using TensorFlow/Keras.
* **Result:** **Catastrophic Overfitting**.
    * **Training Accuracy:** Peaked at **~92%** (Model memorized the training data).
    * **Validation Accuracy:** Stalled at **26.93%** (Model failed to generalize to unseen patterns).
    * **Conclusion:** This explicitly demonstrates that deep learning from scratch requires massive datasets, justifying our move to Transfer Learning.

### 3. ResNet-50 (Transfer Learning)
* **File:** `CNN (ResNet-50 Transfer Learning).ipynb`
* **Approach:** Leveraging a model pre-trained on ImageNet using PyTorch.
* **Technique:** Frozen Backbone (Feature Extractor).
    * Modified Fully Connected Head ($2048 \to 50$ classes).
* **Training:** Fine-tuned for only **10 Epochs**.
* **Result:** **Best Performance (87.48% accuracy)**. The model effectively bridged the semantic gap by reusing learned visual hierarchies.

## 🏆 Performance Comparison

| Model | Method | Training Accuracy | Test/Val Accuracy | Key Observation |
| :--- | :--- | :--- | :--- | :--- |
| **MLP** | Hand-Crafted Features | 98.27% | **81.63%** | Robust baseline using manual features. |
| **Simple CNN** | Trained from Scratch | **~92.00%** | **26.93%** | **Failed:** >60% gap between Train/Val (Overfitting). |
| **ResNet-50** | Transfer Learning | 98.43% | **87.48%** | **Optimal:** Solved the data scarcity issue. |

## 🚀 Installation & Usage

The code is designed to run in **Google Colab**.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/Dog-Breed-Recognition.git](https://github.com/YourUsername/Dog-Breed-Recognition.git)
