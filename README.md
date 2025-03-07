# Comparative Few-Shot Learning for Chest X-Ray Classification

This project presents a comparative study of traditional transfer learning and four popular few-shot learning (FSL) approaches applied to chest X-ray image classification. In this study, we compare the following methods:

- **Original Transfer Learning** (using VGG16 with a custom classifier)
- **Prototypical Networks**

Each method is implemented using a common VGG16 backbone (with pretrained ImageNet weights and frozen convolutional layers) and is evaluated on a dataset of chest X-ray images organized into class directories (e.g., `NORMAL` and `PNEUMONIA`). **Note:** The dataset is not included in this repository but can be downloaded from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

## Overview

Few-shot learning is designed to enable models to learn new concepts quickly with only a few examples. This project uses a conventional transfer learning approach as a baseline and compares it with

**Prototypical Networks:**  
   Computes class prototypes (i.e., the mean embedding for each class) and classifies query images based on their Euclidean distance to these prototypes.

Each approach is trained and evaluated independently from a fresh model initialization to ensure fair comparisons. The project outputs saved model artifacts (HDF5 files) and evaluation metrics (exported to a CSV file).

Models are instantiated, created, and saved within comparative_models.ipynb
Initial approaches for the models are loaded and evaluated within eval.ipynb. An optimized approach for inference within the prototypical network model is then comparatively evaluted within optimized_eval.ipynb.

## Installation and Requirements

Ensure you have Python 3.7 or higher installed. It is recommended to use a virtual environment. Install the required packages with:

```bash
poetry install
