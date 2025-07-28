# 🌿 Yoga Pose Classification 🧘‍♀️

---

### 🎯 Project Goal

To build a model that accurately classifies yoga poses from video data, with improved performance by applying additional techniques.

### 📁 Dataset Overview

- Source: AI-Hub **Yoga Action Dataset**  
  (https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71313)

- Poses used: Advanced-level poses including  
  Navasana, Dhanurasana, Marjaryasana, Utkatasana, Bhujangasana, Adho Mukha Svanasana, Vrikshasana, Virabhadrasana

### 🧪 Input Data Shape

- Train Input: (1939, 33, 30),  Train Labels: (1939, 9)
- Test Input:  (483, 33, 30),  Test Labels:  (483, 9)


---

### ⚙️ Hyperparameters

- Format : (Epochs, Batch Size)
- Epochs : (50, 100)
- Batch Size : (32, 64)

---

### 🌐📊 Global Model Results

| Model       | Best Hyperparameters | Accuracy | Recall | Precision | F1 Score |
|-------------:|----------------------:|----------:|--------:|-----------:|----------:|
| LSTM    | (50, 32)             | 0.9896   | 0.9896 | 0.9898    | 0.9897   |
| GRU     | (50, 32)             | 0.9917   | 0.9917 | 0.9921    | 0.9917   |
| Transformer      | (50, 32)             | __**1.0000**__   | __**1.0000**__ | __**1.0000**__    | __**1.0000**__   |

---

### 📊 LSTM with FedAvg & IPA Results

| Epoch | Method | Best Hyperparameters | Accuracy | Recall | Precision | F1 Score |
|-------:|--------:|----------------------:|----------:|--------:|-----------:|----------:|
| 5     | FedAvg | (100, 32)            | 0.9917   | 0.9917 | 0.9923    | 0.9918   |
| 5     | IPA    | (100, 32)            | 0.9627   | 0.9627 | 0.9711    | 0.9638   |
| 10    | FedAvg | (50, 64)             | 0.9917   | 0.9917 | 0.9926    | 0.9919   |
| 10    | IPA    | (100, 64)            | 0.9855   | 0.9855 | 0.9860    | 0.9856   |
| 15    | FedAvg | (50, 64)             | 0.9855   | 0.9855 | 0.9872    | 0.9857   |
| 15    | IPA    | (50, 64)             | **0.9959**   | **0.9959** | **0.9960**    | **0.9958**   |
| 20    | FedAvg | (100, 64)            | 0.9876   | 0.9876 | 0.9891    | 0.9878   |
| 20    | IPA    | (50, 64)             | 0.9876   | 0.9876 | 0.9893    | 0.9879   |


---

### 📊 GRU with FedAvg & IPA Results

| Epoch | Method | Best Hyperparameters | Accuracy | Recall | Precision | F1 Score |
|-------:|--------:|----------------------:|----------:|--------:|-----------:|----------:|
| 5     | FedAvg | (50, 64)             | **0.9938**   | **0.9938** | 0.9940    | 0.9937   |
| 5     | IPA    | (50, 32)             | 0.9896   | 0.9896 | 0.9902    | 0.9896   |
| 10    | FedAvg | (50, 64)             | 0.9896   | 0.9896 | 0.9918    | 0.9901   |
| 10    | IPA    | (100, 64)            | 0.9689   | 0.9689 | 0.9764    | 0.9700   |
| 15    | FedAvg | (50, 64)             | 0.9917   | 0.9917 | 0.9932    | 0.9920   |
| 15    | IPA    | (50, 32)             | 0.9876   | 0.9876 | 0.9886    | 0.9878   |
| 20    | FedAvg | (50, 32)             | **0.9938**   | **0.9938** | **0.9942**    | **0.9938**   |
| 20    | IPA    | (100, 64)            | 0.9731   | 0.9731 | 0.9762    | 0.9735   |

---

### 📊 Transformer with FedAvg & IPA Results

| Epoch | Method | Best Hyperparameters | Accuracy | Recall | Precision | F1 Score |
|-------:|--------:|----------------------:|----------:|--------:|-----------:|----------:|
| 5     | FedAvg | (50, 32)             | 0.9979   | 0.9979 | 0.9980    | 0.9979   |
| 5     | IPA    | (50, 64)             | 0.9979   | 0.9979 | 0.9980    | 0.9979   |
| 10    | FedAvg | (50, 64)             | **1.0000**   | **1.0000** | **1.0000**    | **1.0000**   |
| 10    | IPA    | (50, 32)             | **1.0000**   | **1.0000** | **1.0000**    | **1.0000**   |
| 15    | FedAvg | (50, 64)             | **1.0000**   | **1.0000** | **1.0000**   | **1.0000**   |
| 15    | IPA    | (50, 32)             | **1.0000**   | **1.0000** | **1.0000**    | **1.0000**   |
| 20    | FedAvg | (100, 64)            | **1.0000**   | **1.0000** | **1.0000**    | **1.0000**   |
| 20    | IPA    | (50, 64)             | **1.0000**   | **1.0000** | **1.0000**    | **1.0000**   |





