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
Input (14×14×4 Quantum Features)
│
├── Flatten
├── Dense(256, L2=0.001) + BatchNorm + ReLU + Dropout(0.4)
├── Dense(128, L2=0.001) + BatchNorm + ReLU + Dropout(0.3)
├── Dense(64, ReLU)
└── Dense(2, Softmax)

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
