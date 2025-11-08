# Quantum-Enhanced Medical Image Diagnostics

## Overview
This project implements a **Hybrid Quantum–Classical Model** to classify images into **Autistic** and **Non-Autistic** categories.  
It combines **Quantum Feature Extraction** using PennyLane with a **Deep Neural Network (DNN)** built in TensorFlow/Keras.

The goal is to demonstrate how **quantum circuits** can enhance feature separability in medical image analysis.

---

## Technologies Used
- **Quantum Framework:** PennyLane (`default.qubit`)
- **Machine Learning:** TensorFlow, Keras
- **Libraries:** NumPy, Scikit-learn, Matplotlib, PIL
- **Environment:** Google Colab (Python 3.10)

---

## Project Workflow
1. **Dataset Loading and Preprocessing**
   - Facial images (Autistic / Non-Autistic)
   - Converted to grayscale, resized (28×28), and normalized
   - Total images used: **600** (balanced classes)

2. **Quantum Feature Extraction**
   - 4-qubit quantum circuit
   - 2-layer `RandomLayers` ansatz for feature transformation
   - Generates 14×14×4 quantum feature maps

3. **Hybrid Model Training**
   - Deep Neural Network with:
     - Dense layers (256 → 128 → 64)
     - Batch Normalization and Dropout
     - L2 regularization
   - Optimizer: Adam (lr = 5e-5)
   - EarlyStopping for optimal convergence

4. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score
   - Confusion Matrix for visual analysis

---

## Model Architecture

| Layer (Type) | Output Shape | Parameters | Description |
|:--|:--:|:--|
| **InputLayer** | (14, 14, 4) | 0 | Input quantum feature map (from 4-qubit circuit) |
| **Flatten** | (784) | 0 | Converts 2D quantum features into 1D vector |
| **Dense (256)** | (256) | 200,960 | Fully connected layer with L2 regularization |
| **BatchNormalization** | (256) | 1,024 | Normalizes activations for stable training |
| **ReLU Activation** | (256) | 0 | Introduces non-linearity |
| **Dropout (0.4)** | (256) | 0 | Prevents overfitting by randomly dropping neurons |
| **Dense (128)** | (128) | 32,896 | Second hidden layer with L2 regularization |
| **BatchNormalization** | (128) | 512 | Normalizes activations |
| **ReLU Activation** | (128) | 0 | Non-linear activation |
| **Dropout (0.3)** | (128) | 0 | Regularization layer |
| **Dense (64)** | (64) | 8,256 | Third hidden layer for deeper feature extraction |
| **ReLU Activation** | (64) | 0 | Non-linear transformation |
| **Dense (2)** | (2) | 130 | Output layer with Softmax for binary classification |
| **Total Parameters** | — | **243,778** | Trainable model parameters |

---

### Summary
- **Optimizer:** Adam (learning rate = 5e-5)  
- **Loss Function:** Sparse Categorical Crossentropy  
- **Metrics:** Accuracy  
- **Regularization:** L2 (λ = 0.001) + Dropout (0.3–0.4)  
- **EarlyStopping:** Enabled for validation loss  
- **Model Type:** Quantum–Classical Hybrid Neural Network

---

## Results
| Metric | Value |
|:--|:--:|
| **Accuracy** | 70.0 % |
| **Precision (Autistic)** | 0.73 |
| **Recall (Autistic)** | 0.66 |
| **Precision (Non-Autistic)** | 0.67 |
| **Recall (Non-Autistic)** | 0.74 |
| **Macro F1-Score** | 0.70 |

 Balanced performance shows that quantum features improved the model’s ability to distinguish between both classes.

---

## Visualization
- Displayed 10 random images (5×2 grid) to confirm preprocessing.
- Visualized **quantum feature maps** for channels 0–2 to understand feature transformation.
- Plotted training vs validation accuracy and loss.

---

## Conclusion
- Increasing dataset size from **300 → 600** images improved accuracy from ~55% → **70%**.  
- Quantum feature extraction enhanced feature representation and model generalization.  
- The hybrid approach demonstrates strong potential for **quantum-assisted medical diagnostics**.

---

## Authors
- **Abhinav Marlingaplar** — 2023BCD0013  
- **Bhaskara Akshay Sriram** — 2023BCD0015  

Indian Institute of Information Technology, Kottayam  
B.Tech in Computer Science (AI & Data Science)

---

## License
This project is released under the **MIT License**.  
Feel free to use or modify it for academic and research purposes.
