

---

# 🎂 Deep Age Estimator: Facial Age Prediction using ResNet18

This project implements a deep learning model to estimate a person's age from facial images. By leveraging **Transfer Learning** and the **ResNet18** architecture, the model maps visual features to a continuous numerical age value.

## 📝 Project Overview

Age estimation is a complex regression task in Computer Vision. This project demonstrates how pre-trained models on ImageNet can be fine-tuned to achieve high accuracy in specialized tasks like facial analysis.

### Technical Stack:

* **Framework:** PyTorch
* **Model:** ResNet18 (Pre-trained)
* **Task:** Regression
* **Loss Function:** `SmoothL1Loss` (Robust to outliers)
* **Optimization:** Adam Optimizer ()

## 🛠️ Implementation Highlights

1. **Transfer Learning:** Modified the ResNet18 head by replacing the final FC layer with a single output neuron.
2. **Fine-tuning:** Set all parameters to trainable to capture specific facial aging patterns.
3. **Data Pipeline:** Custom PyTorch Dataset and DataLoaders for efficient image preprocessing and batching.

## 📊 Results & Performance

The model was trained for **5 epochs**, showing rapid convergence:

* **Initial Train Loss:** 15.31
* **Final Train Loss:** 4.26
* **Test Accuracy:** Achieved a **Mean Absolute Error (MAE) of 4.94 years**.

### Visual Analysis

The scatter plot of **True Age vs. Predicted Age** shows a strong linear correlation, with most predictions aligning closely with the  line. This confirms the model's reliability across various age groups.

## 🚀 How to Run

1. Clone the repository.
2. Install dependencies:
```bash
pip install torch torchvision matplotlib numpy pandas

```


3. Run the Jupyter notebook `notebookb8f139c015.ipynb`.

---

