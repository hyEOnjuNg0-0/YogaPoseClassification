# 🌿 Yoga Pose Classification (요가 동작 분류) 🧘‍♀️


### 🎯 Project Goal

> To build a deep learning model that accurately classifies yoga poses from video data, with improved performance by applying additional techniques.

> 비디오 데이터를 기반으로 요가 동작을 정확하게 분류하는 딥러닝 모델을 구축하고, 추가 기법을 적용해 성능을 향상시키고자 한다.

### 📁 Dataset Overview

- Source: AI-Hub **Yoga Action Dataset**  
  (https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71313)

- Poses used: Advanced-level poses including  
  Navasana, Dhanurasana, Marjaryasana, Utkatasana, Bhujangasana, Adho Mukha Svanasana, Vrikshasana, Virabhadrasana

### 🧪 Input Data Shape

- Train Input: (1939, 33, 30),  Train Labels: (1939, 9)
- Test Input:  (483, 33, 30),  Test Labels:  (483, 9)


---

### ⚙️ Hyperparameters for NestedCV

- Format : (Epochs, Batch Size)
- Epochs : (50, 100)
- Batch Size : (32, 64)

---
### 🛠️ Additional Techniques
- **FedAvg (Federated Averaging)**
  
  > Each local model is trained independently, after which the weights from the local models are averaged. The resulting averaged weights are then transmitted to the global model for updating.

- **IPA (Iterative Parameter Averaging)**
  
  > Two local models are initially trained and their weights are averaged. The averaged weights are then applied to the next two local models, and this process is repeated iteratively.

---
### 🌐📊 Global Model Results

| Model       | Best Hyperparameters | Accuracy | Recall | Precision | F1 Score |
|-------------:|----------------------:|----------:|--------:|-----------:|----------:|
| LSTM    | (50, 32)             | 0.9896   | 0.9896 | 0.9898    | 0.9897   |
| GRU     | (50, 32)             | 0.9917   | 0.9917 | 0.9921    | 0.9917   |
| Transformer      | (50, 32)             | **_1.0000_**   | **_1.0000_** | **_1.0000_**   | _**1.0000**_   |

---

### 📊 LSTM with FedAvg & IPA Results

| Local Model Num | Method | Best Hyperparameters | Accuracy | Recall | Precision | F1 Score |
|-------:|--------:|----------------------:|----------:|--------:|-----------:|----------:|
| 5     | FedAvg | (100, 32)            | 0.9917   | 0.9917 | 0.9923    | 0.9918   |
| 5     | IPA    | (100, 32)            | 0.9627   | 0.9627 | 0.9711    | 0.9638   |
| 10    | FedAvg | (50, 64)             | 0.9917   | 0.9917 | 0.9926    | 0.9919   |
| 10    | IPA    | (100, 64)            | 0.9855   | 0.9855 | 0.9860    | 0.9856   |
| 15    | FedAvg | (50, 64)             | 0.9855   | 0.9855 | 0.9872    | 0.9857   |
| 15    | IPA    | (50, 64)             | **_0.9959_**   | **_0.9959_** | **_0.9960_**    | **_0.9958_**   |
| 20    | FedAvg | (100, 64)            | 0.9876   | 0.9876 | 0.9891    | 0.9878   |
| 20    | IPA    | (50, 64)             | 0.9876   | 0.9876 | 0.9893    | 0.9879   |


---

### 📊 GRU with FedAvg & IPA Results

| Local Model Num | Method | Best Hyperparameters | Accuracy | Recall | Precision | F1 Score |
|-------:|--------:|----------------------:|----------:|--------:|-----------:|----------:|
| 5     | FedAvg | (50, 64)             | **_0.9938_**   | **_0.9938_** | 0.9940    | 0.9937   |
| 5     | IPA    | (50, 32)             | 0.9896   | 0.9896 | 0.9902    | 0.9896   |
| 10    | FedAvg | (50, 64)             | 0.9896   | 0.9896 | 0.9918    | 0.9901   |
| 10    | IPA    | (100, 64)            | 0.9689   | 0.9689 | 0.9764    | 0.9700   |
| 15    | FedAvg | (50, 64)             | 0.9917   | 0.9917 | 0.9932    | 0.9920   |
| 15    | IPA    | (50, 32)             | 0.9876   | 0.9876 | 0.9886    | 0.9878   |
| 20    | FedAvg | (50, 32)             | **_0.9938_**   | **_0.9938_** | **_0.9942_**    | **_0.9938_**   |
| 20    | IPA    | (100, 64)            | 0.9731   | 0.9731 | 0.9762    | 0.9735   |

---

### 📊 Transformer with FedAvg & IPA Results

| Local Model Num | Method | Best Hyperparameters | Accuracy | Recall | Precision | F1 Score |
|-------:|--------:|----------------------:|----------:|--------:|-----------:|----------:|
| 5     | FedAvg | (50, 32)             | 0.9979   | 0.9979 | 0.9980    | 0.9979   |
| 5     | IPA    | (50, 64)             | 0.9979   | 0.9979 | 0.9980    | 0.9979   |
| 10    | FedAvg | (50, 64)             | **_1.0000_**   | **_1.0000_** | **_1.0000_**    | **_1.0000_**   |
| 10    | IPA    | (50, 32)             | **_1.0000_**   | **_1.0000_** | **_1.0000_**    | **_1.0000_**   |
| 15    | FedAvg | (50, 64)             | **_1.0000_**   | **_1.0000_** | **_1.0000_**    | **_1.0000_**   |
| 15    | IPA    | (50, 32)             | **_1.0000_**   | **_1.0000_** | **_1.0000_**    | **_1.0000_**   |
| 20    | FedAvg | (100, 64)            | **_1.0000_**   | **_1.0000_** | **_1.0000_**    | **_1.0000_**   |
| 20    | IPA    | (50, 64)             | **_1.0000_**   | **_1.0000_** | **_1.0000_**    | **_1.0000_**   |





