# Multi-Class Classification of Pigmented Skin Lesions

**Final Project | Pengolahan Sinyal Medis (Biomedical Signal Processing)** Department of Biomedical Engineering  
*June 13, 2025*

</div>

---

### **Group 10 Members:**
* **Diniya Fakhriza Aulia Putri Zahra** (2206820314)
* **Hasan Abdul Lathif** (2206815503)
* **Josephine Sere Emmanuela Tamba** (2206060706)

---

## 1. Abstract

This repository contains the final project for the BSIP course: an end-to-end system for the multi-class classification of common pigmented skin lesions, designed to aid in the early detection of skin cancer. The system is trained on the **HAM10000 dataset** and implements two distinct deep learning models for comparison: a baseline Convolutional Neural Network (CNN) and an advanced, fine-tuned **DenseNet121** transfer learning model. The project includes a full pipeline from data preprocessing and exploratory analysis to model training and a fully interactive Graphical User Interface (GUI) built with Streamlit for real-time inference on new images.

## 2. Background

The accurate diagnosis of pigmented skin lesions is a critical task for the early prevention and treatment of skin cancer. A significant challenge in clinical practice is distinguishing malignant lesions (such as melanoma) from various benign ones. While dermoscopy, a non-invasive imaging technique, has proven to enhance diagnostic accuracy compared to naked-eye examination, its effectiveness can be limited by clinical constraints and inter-observer variability among medical professionals. The recent growth of large, public datasets of digital dermoscopic images, combined with advancements in deep learning, presents a powerful opportunity to develop automated classification systems to support clinicians and improve diagnostic outcomes.

## 3. Problem Statement

Automated diagnostic models trained on small or non-diverse datasets often exhibit high performance in controlled experimental settings but fail to generalize to the variety and complexity of real-world clinical conditions. Furthermore, much of the existing research has focused on binary classification (e.g., melanoma vs. non-melanoma), whereas a clinician must differentiate between a wide spectrum of lesion types. This project addresses the need for a robust **multi-category classification system** for dermoscopic images that can handle the diversity of lesions encountered in daily practice.

## 4. Proposed Solution & Methodology

This project implements a deep learning-based classification system using the comprehensive **HAM10000 dataset**, which features over 10,000 dermoscopic images across seven distinct diagnostic categories. The methodology follows a structured machine learning pipeline:

1.  **Exploratory Data Analysis (EDA):** The dataset was first analyzed to understand class distributions, identify challenges such as significant class imbalance, and inspect the metadata.

2.  **Data Preprocessing & Augmentation:** Images are resized and normalized for model consumption. To create a more robust training process, on-the-fly data augmentation (random rotations, zooms, flips) is applied to the training images, effectively creating synthetic variations of the data.

3.  **Model Architecture:** Two models were developed for comparison:
    * **Baseline Model:** A custom-built Convolutional Neural Network (CNN) to establish a performance baseline.
    * **Advanced Model:** A `DenseNet121` architecture pre-trained on ImageNet, implemented using transfer learning.

4.  **Training Strategy (Fine-Tuning):** To address the class imbalance, a **class weighting** strategy was employed during training, penalizing the model more for misclassifying samples from minority classes. The DenseNet121 model was **fine-tuned** by first training a custom classification head and then unfreezing the top layers of the base model to continue training with a low learning rate.

5.  **Deployment:** The final trained models are integrated into an interactive GUI built with **Streamlit**, allowing for real-time classification of user-uploaded images.

## 5. Repository Contents

This repository contains all necessary files to replicate the project. The large model files are tracked using Git LFS.

* **`README.md`**: This file, providing a complete overview of the project.
* **`skin_lesion_classification_notebook.ipynb`**: The Google Colab notebook containing the full code for data loading, EDA, preprocessing, model training, and evaluation.
* **`app.py`**: The Python script for the Streamlit GUI.
* **`requirements.txt`**: A list of all necessary Python libraries to run the project.
* **`.gitattributes`**: A configuration file that tells Git to use LFS for the large model files.
* **`.h5` files**: The trained model files (`baseline_model.h5` and `skin_cancer_densenet121_finetuned.h5`), tracked using Git LFS, are located in the main directory.

## 6. Setup and Installation

To run this project locally, please follow the steps below.

**Prerequisites:**
* Python 3.9+
* Git
* [Git LFS (Large File Storage)](https://git-lfs.github.com/)

### Step 1: Clone the Repository

Clone this repository to your local machine. Git LFS will automatically detect the `.h5` files and download them during the cloning process.

```bash
git clone [https://github.com/HasanLathif/PSB-Final-Project-Group10.git](https://github.com/HasanLathif/PSB-Final-Project-Group10.git)
cd PSB-Final-Project-Group10
```
> **Note:** If the large model files do not download automatically, you can fetch them manually by running `git lfs pull` inside the repository folder.

### Step 2: Install Dependencies

It is highly recommended to use a virtual environment to avoid conflicts with other projects.
```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required libraries from the requirements file
pip install -r requirements.txt
```

## 7. How to Run the GUI

Once the setup is complete and the models have been downloaded via Git LFS, launch the Streamlit application with the following command:

```bash
streamlit run app.py
```

This will open the interactive GUI in your default web browser, where you can upload an image for classification.
