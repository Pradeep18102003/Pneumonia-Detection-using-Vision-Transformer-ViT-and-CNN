# 🩺 Pneumonia Detection using Vision Transformer (ViT) and CNN

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)
![Model Accuracy](https://img.shields.io/badge/ViT_Accuracy-100%25-success.svg)
![Model Accuracy](https://img.shields.io/badge/CNN_Accuracy-85.9%25-yellow.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📘 Overview
This project focuses on **detecting Pneumonia from Chest X-ray images** using two deep learning models — a **Convolutional Neural Network (CNN)** and a **Vision Transformer (ViT)**.  
The goal is to explore and compare the performance of traditional CNN-based architectures against state-of-the-art transformer-based vision models in a medical image classification task.

---

## 🧠 Models Implemented
### 1. Convolutional Neural Network (CNN)
- Based on **ResNet34** architecture.  
- Trained using transfer learning with fine-tuning on the Pneumonia dataset.  
- Achieved **85.9% accuracy** and **F1-score of 0.8965**.

### 2. Vision Transformer (ViT)
- Implemented using **`timm`** pretrained ViT model.  
- Captures long-range dependencies in X-ray images, improving generalization.  
- Achieved **100% validation accuracy** and **F1-score of 1.0**, outperforming the CNN baseline.

---

## 📊 Dataset
- **Dataset Used:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes:**  
  - `NORMAL`  
  - `PNEUMONIA`  
- The dataset is divided into `train`, `test`, and `validation` folders.

---

## ⚙️ Implementation Steps
1. **Data Preprocessing:**  
   - Image resizing, normalization, and augmentation using `torchvision.transforms`.  
   - Custom PyTorch `Dataset` and `DataLoader` created for efficient batching.  
2. **Model Training:**  
   - Optimizer: Adam / SGD  
   - Scheduler: CosineAnnealingLR  
   - Loss: CrossEntropyLoss  
3. **Evaluation Metrics:**  
   - Accuracy, Precision, Recall, F1-Score, and Confusion Matrix  

---

## 🚀 Results Summary
| Model | Accuracy | Precision | Recall | F1-Score |
|:------|:----------:|:-----------:|:---------:|:----------:|
| CNN (ResNet34) | **85.9%** | 0.8283 | 0.9769 | 0.8965 |
| Vision Transformer (ViT) | **100%** | 1.0000 | 1.0000 | 1.0000 |

> ✅ The **ViT model** outperformed the CNN by a significant margin, demonstrating the power of transformer-based architectures for medical image classification.

---

## 🧩 Repository Structure
```
📂 Pneumonia Detection using Vision Transformer (ViT) and CNN
│── 📂 notebook
│   └── pneumonia-detection-using-vit-and-cnn.ipynb  # Model Training
│── README.md  # Project documentation
```

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Frameworks:** PyTorch, timm, torchvision  
- **Libraries:** scikit-learn, matplotlib, seaborn, numpy, pandas  

---

## 🧾 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/pneumonia-detection-using-vit-and-cnn.git
   cd pneumonia-detection-using-vit-and-cnn
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the notebook:
   ```bash
   jupyter notebook pneumonia-detection-using-vit-and-cnn.ipynb

## 🙌 Acknowledgements:
[Kaggle Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## License
This project is licensed under the [MIT License](LICENSE).

## 📧 Contact
- If you have any questions or feedback, feel free to reach out via email: pradeep18kumar10@gmail.com  
- LinkedIn: [Pradeep Kumar](https://www.linkedin.com/in/pradeep-kumar-bba090320/)
