# Yoga Pose Classification

최소 프레임 :

D:\yoga\yoga_video\tree\tree (18).mp4 - 53 frames

최대 프레임 :

D:\yoga\yoga_video\cat\cat (36).mp4 - 232 frames

---------------------------------------------------------------------------------------
Shape Of Padded Train Input Data : (277, 232, 30)

Shape Of Padded Train Labels Data : (277, 232)

Shape Of Padded Test Input Data : (69, 232, 30)

Shape Of Padded Test Labels Data : (69, 232)


Train input shape: (1939, 33, 30), Train label shape: (1939,)

Test input shape: (483, 33, 30), Test label shape: (483,)


Shape Of Encoded Labels : (1939, 9)

Shape Of Encoded Labels : (483, 9)


---------------------------------------------------------------------------------------
## LSTM

Best Hyperparameters for LSTM: (50, 32)

---------- LSTM Global Model Evaluation ----------
Accuracy : 0.9896
Recall: 0.9896
Precision: 0.9898
F1 Score: 0.9897

시작 시각: 2025-07-14 14:10:08
끝 시각: 2025-07-14 14:35:59

---------------------------------------------------------------------------------------
## GRU

Best Hyperparameters for GRU: (50, 32)

---------- GRU Global Model Evaluation ----------
Accuracy : 0.9917
Recall: 0.9917
Precision: 0.9921
F1 Score: 0.9917

시작 시각: 2025-07-14 14:53:06
끝 시각: 2025-07-14 15:18:11


---------------------------------------------------------------------------------------
## Transformer

Best Hyperparameters for TF: (50, 32)

---------- TF Global Model Evaluation ----------
Accuracy : 1.0000
Recall: 1.0000
Precision: 1.0000
F1 Score: 1.0000

시작 시각: 2025-07-14 15:21:41
끝 시각: 2025-07-14 16:00:54


---------------------------------------------------------------------------------------
## LSTM with FedAvg and IPA

Best Hyperparameters for LSTM FA: (100, 32)
---------- LSTM 5 FedAvg Evaluation ----------
Accuracy : 0.9917
Recall: 0.9917
Precision: 0.9923
F1 Score: 0.9918

Best Hyperparameters for LSTM IPA: (100, 32)
---------- LSTM 5 IPA Evaluation ----------
Accuracy : 0.9627
Recall: 0.9627
Precision: 0.9711
F1 Score: 0.9638

시작 시각: 2025-07-15 15:22:35
끝 시각: 2025-07-15 16:20:26

--
Best Hyperparameters for LSTM FA: (50, 64)

---------- LSTM 10 FedAvg Evaluation ----------
Accuracy : 0.9917
Recall: 0.9917
Precision: 0.9926
F1 Score: 0.9919

Best Hyperparameters for LSTM IPA: (100, 64)

---------- LSTM 10 IPA Evaluation ----------
Accuracy : 0.9855
Recall: 0.9855
Precision: 0.9860
F1 Score: 0.9856

시작 시각: 2025-07-15 16:23:08
끝 시각: 2025-07-15 17:38:12

--
Best Hyperparameters for LSTM FA: (50, 64)

---------- LSTM 15 FedAvg Evaluation ----------
Accuracy : 0.9855
Recall: 0.9855
Precision: 0.9872
F1 Score: 0.9857

Best Hyperparameters for LSTM IPA: (50, 64)

---------- LSTM 15 IPA Evaluation ----------
Accuracy : 0.9959
Recall: 0.9959
Precision: 0.9960
F1 Score: 0.9958

시작 시각: 2025-07-15 17:38:59
끝 시각: 2025-07-15 18:46:42

--
Best Hyperparameters for LSTM FA: (100, 64)
---------- LSTM 20 FedAvg Evaluation ----------
Accuracy : 0.9876
Recall: 0.9876
Precision: 0.9891
F1 Score: 0.9878

Best Hyperparameters for LSTM IPA: (50, 64)
---------- LSTM 20 IPA Evaluation ----------
Accuracy : 0.9876
Recall: 0.9876
Precision: 0.9893
F1 Score: 0.9879

시작 시각: 2025-07-15 19:06:30
끝 시각: 2025-07-15 20:23:58

---------------------------------------------------------------------------------------
## GRU with FedAvg and IPA

Best Hyperparameters for GRU FA: (50, 64)
---------- GRU 5 FedAvg Evaluation ----------
Accuracy : 0.9938
Recall: 0.9938
Precision: 0.9940
F1 Score: 0.9937

Best Hyperparameters for GRU IPA: (50, 32)
---------- GRU 5 IPA Evaluation ----------
Accuracy : 0.9896
Recall: 0.9896
Precision: 0.9902
F1 Score: 0.9896

시작 시각: 2025-07-15 20:58:22
끝 시각: 2025-07-15 21:51:36

--
Best Hyperparameters for GRU FA: (50, 64)
---------- GRU 10 FedAvg Evaluation ----------
Accuracy : 0.9896
Recall: 0.9896
Precision: 0.9918
F1 Score: 0.9901

Best Hyperparameters for GRU IPA: (100, 64)
---------- GRU 10 IPA Evaluation ----------
Accuracy : 0.9689
Recall: 0.9689
Precision: 0.9764
F1 Score: 0.9700

시작 시각: 2025-07-15 22:10:29
끝 시각: 2025-07-15 23:08:29

--
Best Hyperparameters for GRU FA: (50, 64)
---------- GRU 15 FedAvg Evaluation ----------
Accuracy : 0.9917
Recall: 0.9917
Precision: 0.9932
F1 Score: 0.9920

Best Hyperparameters for GRU IPA: (50, 32)
---------- GRU 15 IPA Evaluation ----------
Accuracy : 0.9876
Recall: 0.9876
Precision: 0.9886
F1 Score: 0.9878

시작 시각: 2025-07-16 17:58:04
끝 시각: 2025-07-16 18:54:53

--
Best Hyperparameters for GRU FA: (50, 32)
---------- GRU 20 FedAvg Evaluation ----------
Accuracy : 0.9938
Recall: 0.9938
Precision: 0.9942
F1 Score: 0.9938

Best Hyperparameters for GRU IPA: (100, 64)
---------- GRU 20 IPA Evaluation ----------
Accuracy : 0.9731
Recall: 0.9731
Precision: 0.9762
F1 Score: 0.9735

시작 시각: 2025-07-16 18:59:11
끝 시각: 2025-07-16 19:59:54

---------------------------------------------------------------------------------------
## Transformer with FedAvg and IPA

Best Hyperparameters for TF FA: (50, 32)
---------- TF 5 FedAvg Evaluation ----------
Accuracy : 0.9979
Recall: 0.9979
Precision: 0.9980
F1 Score: 0.9979

Best Hyperparameters for TF IPA: (50, 64)
---------- TF 5 IPA Evaluation ----------
Accuracy : 0.9979
Recall: 0.9979
Precision: 0.9980
F1 Score: 0.9979

시작 시각: 2025-07-16 20:20:02
끝 시각: 2025-07-16 21:24:24

--
Best Hyperparameters for TF FA: (50, 64)
---------- TF 10 FedAvg Evaluation ----------
Accuracy : 1.0000
Recall: 1.0000
Precision: 1.0000
F1 Score: 1.0000

Best Hyperparameters for TF IPA: (50, 32)
---------- TF 10 IPA Evaluation ----------
Accuracy : 1.0000
Recall: 1.0000
Precision: 1.0000
F1 Score: 1.0000

시작 시각: 2025-07-16 21:31:31
끝 시각: 2025-07-16 22:38:13

--
Best Hyperparameters for TF FA: (50, 64)
---------- TF 15 FedAvg Evaluation ----------
Accuracy : 1.0000
Recall: 1.0000
Precision: 1.0000
F1 Score: 1.0000

Best Hyperparameters for TF IPA: (50, 32)
---------- TF 15 IPA Evaluation ----------
Accuracy : 1.0000
Recall: 1.0000
Precision: 1.0000
F1 Score: 1.0000

시작 시각: 2025-07-16 23:02:12
끝 시각: 2025-07-17 00:17:16

--
Best Hyperparameters for TF FA: (100, 64)
---------- TF 20 FedAvg Evaluation ----------
Accuracy : 1.0000
Recall: 1.0000
Precision: 1.0000
F1 Score: 1.0000

Best Hyperparameters for TF IPA: (50, 64)
---------- TF 20 IPA Evaluation ----------
Accuracy : 1.0000
Recall: 1.0000
Precision: 1.0000
F1 Score: 1.0000

시작 시각: 2025-07-17 01:29:47
끝 시각: 2025-07-17 02:45:29






