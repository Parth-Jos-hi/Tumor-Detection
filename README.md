🧠 Brain Tumor MRI Analysis using Convolutional Neural Networks
📌 Project Overview

This project is a learning-focused deep learning study that explores how Convolutional Neural Networks (CNNs) learn, interpret, and fail when applied to brain MRI images.

The goal is not medical diagnosis, but to understand CNN behavior in a challenging, real-world vision task involving subtle, localized abnormalities.

🎯 Objectives

Build a CNN from scratch for brain MRI abnormality classification

Understand how CNNs learn spatial and texture-based features

Analyze where the model focuses while making predictions

Study model sensitivity and failure cases

Learn interpretability techniques used in real ML systems

🧠 Problem Statement

Brain tumors in MRI images are often:

Small

Localized

Visually similar to normal tissue

This makes them a difficult vision problem and an excellent case study for learning how CNNs:

Extract hierarchical features

Use spatial information

Make and justify predictions

The task is formulated as a binary classification problem:

Yes → Tumor present

No → No tumor present

📂 Dataset

Dataset: Brain MRI Images for Brain Tumor Detection

Source: Kaggle

Image type: Grayscale MRI slices

Classes:

yes (tumor)

no (no tumor)

This dataset is used strictly for educational and research purposes.

🏗️ Project Structure
brain_tumor_cnn/
│
├── data/
│   └── brain_tumor_dataset/
│       ├── yes/
│       └── no/
│
├── notebooks/
│   └── exploration.ipynb
│
├── model.py
├── train.py
├── analyze.py
├── app.py
└── README.md

🧪 Methodology
1️⃣ Data Exploration

Visual inspection of MRI images

Class distribution analysis

Image size and intensity analysis

(done in Jupyter Notebook)

2️⃣ CNN Model (From Scratch)

Multiple convolution layers

ReLU activations

Max pooling

Fully connected layers

Dropout for regularization

No pretrained models are used.

3️⃣ Training & Evaluation

Binary Cross-Entropy loss

Adam optimizer

Accuracy and recall as evaluation metrics

Confusion matrix analysis

Special focus is placed on false negatives, as missing abnormalities is critical in medical contexts.

4️⃣ Model Interpretation & Analysis

To understand how the CNN makes decisions, the following techniques are used:

🔹 Occlusion Sensitivity

Parts of the image are masked

Changes in prediction confidence are observed

Helps identify spatial dependency

🔹 Grad-CAM

Gradient-based visualization

Highlights regions influencing predictions

Provides insight into feature importance

<img width="1495" height="584" alt="Screenshot 2026-01-02 115530" src="https://github.com/user-attachments/assets/ab5c236f-b5a7-4a88-b8b7-d7e46761df34" />

5️⃣ Interactive Visualization (Optional)

A minimal Streamlit app is included to:

Upload MRI images

View predictions and confidence scores

Visualize Grad-CAM heatmaps

This is used strictly for model probing and understanding, not deployment.

⚠️ Important Notes & Limitations

This project does NOT perform medical diagnosis

Results should not be used for clinical decisions

Dataset size is limited

Tumor localization is approximate, not pixel-accurate

📈 Key Learnings

How CNNs learn hierarchical features in medical images

Why spatial context matters in vision models

How interpretability tools reveal model behavior

How CNNs fail under noise, occlusion, and low contrast

Why accuracy alone is insufficient for evaluation

🚀 Future Extensions

Tumor localization

Segmentation (U-Net style)

Multi-class tumor classification

Robustness testing across MRI sequences

🧾 Conclusion

This project serves as a foundational deep learning study that transforms CNNs from black-box predictors into understandable, analyzable systems.

It prioritizes learning, interpretability, and responsible ML practice over flashy results.
