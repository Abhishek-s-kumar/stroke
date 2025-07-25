# Multimodal Stroke Image Classification

This project implements a deep learning model to classify stroke from medical images, leveraging a hybrid ResNet50 and Vision Transformer (ViT) architecture. The solution includes data download, preparation, model training, comprehensive evaluation, and advanced visualization techniques like Grad-CAM for interpretability.

---

## 🚀 Project Overview

The goal of this project is to accurately detect stroke in medical images (both BT and MR modalities). A key aspect is the use of a **hybrid deep learning model** combining the strengths of Convolutional Neural Networks (ResNet50) for local feature extraction and Vision Transformers (ViT) for global context understanding.

The pipeline covers:
* **Dataset Download & Setup**: Utilizing `kagglehub` to fetch the dataset.
* **Data Preprocessing**: Custom `Dataset` and `DataLoader` for efficient image handling and augmentation.
* **Model Architecture**: A custom `ResNet50_ViT` model that concatenates features from both backbones.
* **Training & Evaluation**: Training the model with standard deep learning practices and evaluating performance using various metrics.
* **Performance Metrics**: Detailed analysis including accuracy, confusion matrix, ROC curve, Precision-Recall curve, and class-wise/modality-wise breakdowns.
* **Interpretability with Grad-CAM**: Visualizing the model's focus areas using enhanced Grad-CAM techniques for both ResNet and ViT components.
* **Results Export**: Saving detailed predictions and summary statistics to CSV files.

---

## 🛠️ Installation and Setup

To run this project, you'll need to set up your Python environment and install the necessary libraries.

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repo-link>
    cd <your-repo-name>
    ```

2.  **Install required packages:**
    All necessary packages can be installed using `pip`. The `grad-cam` library is specifically handled to ensure it's installed if not present.

    ```bash
    !pip -q install kaggle timm einops albumentations==1.4.6 torchmetrics wandb kagglehub grad-cam pandas scikit-learn
    ```
    *Note: The `albumentations` version is pinned to `1.4.6` for compatibility.*

---

## 📊 Dataset

The dataset used is the **"Multimodal Stroke Image Dataset"** from Kaggle, downloaded via `kagglehub`.

The dataset structure is organized into `train` and `test` (referred to as `val` in the code) directories, with subfolders for `1- Stroke` and `2- Control` (for training) and specific modalities like `strokeBT`, `strokeMR`, `normalBT`, `normalMR` for testing.

````

deep/
test/
strokeMR/
normalBT/
strokeBT/
normalMR/
train/
1- Stroke/
2- Control/

```

---

## 🧠 Model Architecture

The core of this project is the `ResNet50_ViT` model, which is a hybrid deep learning architecture:

* **ResNet50**: A pre-trained ResNet50 model (from `timm`) is used as a feature extractor. Its global pooling and final classification layers are replaced with `nn.Identity()`.
* **Vision Transformer (ViT)**: A pre-trained `vit_base_patch16_224` model (from `timm`) is also used as a feature extractor. Its classification head is replaced with `nn.Identity()`.
* **Feature Concatenation**: Features extracted from both ResNet (after adaptive average pooling) and ViT (the CLS token) are concatenated.
* **Custom Classifier**: A sequential `nn.Linear` block with `ReLU` activation and `Dropout` layers is added on top of the concatenated features for binary classification (Stroke vs. Normal).

---

## 📈 Training and Evaluation

### Training Configuration

* **Image Size**: 224x224 pixels
* **Batch Size**: 32
* **Epochs**: 35
* **Loss Function**: `nn.CrossEntropyLoss`
* **Optimizer**: `torch.optim.AdamW` with a learning rate of `3e-4` and `weight_decay` of `1e-4`.
* **Metrics**: Binary Accuracy and AUROC (Area Under the Receiver Operating Characteristic Curve) using `torchmetrics`.

### Training Progress

The training process logs per-epoch training loss, validation loss, accuracy, and AUROC. Model checkpoints are saved every 10 epochs.

### Evaluation Metrics

The model's performance is comprehensively evaluated on the validation dataset using:

* **Overall Accuracy**
* **Classification Report**: Precision, Recall, F1-score for "Normal" and "Stroke" classes.
* **Confusion Matrix**: Visualizing True Positives, True Negatives, False Positives, and False Negatives.
* **ROC Curve and AUROC**: Assessing the model's ability to distinguish between classes across various thresholds.
* **Precision-Recall Curve and Average Precision**: Particularly useful for imbalanced datasets.
* **Prediction Probability Distribution**: Histograms showing the distribution of predicted probabilities for both true classes.

---

## 🎨 Visualizations

The project includes powerful visualizations to understand the model's performance and decision-making:

* **Training Curves**: Plots of training/validation loss, validation accuracy, and validation AUROC over epochs.
* **Confusion Matrix**: A direct visual representation of classification performance.
* **ROC and PR Curves**: Standard plots to assess classifier quality.
* **Grad-CAM Visualizations (Enhanced)**:
    * **ResNet Features**: Shows heatmaps highlighting regions of interest identified by the ResNet backbone.
    * **ViT Attention**: Visualizes the attention paid by the Vision Transformer to different image patches.
    * **Combined View**: A fusion of ResNet CAM and ViT attention maps, providing a holistic understanding of the hybrid model's focus.
    * **Component Analysis Summary**: A detailed breakdown for selected samples, comparing how ResNet and ViT focus, along with their contribution norms and prediction details.

---

## 🔬 Analysis by Imaging Modality

The project also provides a breakdown of performance metrics (accuracy, AUROC, AP, and classification report) specifically for **Brain Tomography (BT)** and **Magnetic Resonance (MR)** images, allowing for insights into the model's performance across different imaging types.

---

## 💾 Saved Results

Upon completion, the project generates the following output files:

* `stroke_classification_results.csv`: A detailed CSV file containing per-image predictions, true labels, predicted probabilities, and derived information like `image_type` and `correct_prediction`.
* `summary_statistics.csv`: A CSV file summarizing overall performance metrics (total images, accuracy, AUROC, AP, and confusion matrix counts).
* `model_epoch_*.pth`: Model checkpoints saved during training.

These files provide a comprehensive record of the model's performance and can be used for further analysis or deployment.

---
