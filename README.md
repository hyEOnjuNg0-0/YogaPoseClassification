# Yoga Pose Classification

최소 프레임 :
D:\yoga\yoga_video\tree\tree (18).mp4 - 53 frames
최대 프레임 :
D:\yoga\yoga_video\cat\cat (36).mp4 - 232 frames

Shape Of Padded Train Input Data : (277, 232, 30)
Shape Of Padded Train Labels Data : (277, 232)
Shape Of Padded Test Input Data : (69, 232, 30)
Shape Of Padded Test Labels Data : (69, 232)

Train input shape: (1939, 33, 30), Train label shape: (1939,)
Test input shape: (483, 33, 30), Test label shape: (483,)

Shape Of Encoded Labels : (1939, 9)
Shape Of Encoded Labels : (483, 9)


---------------------------------------------------------------------------------------
< LSTM >
Best Hyperparameters for LSTM: (50, 32)
---------- LSTM Global Model Evaluation ----------
Accuracy : 0.9896
Recall: 0.9896
Precision: 0.9898
F1 Score: 0.9897

시작 시각: 2025-07-14 14:10:08
끝 시각: 2025-07-14 14:35:59

---------------------------------------------------------------------------------------
< GRU >
Best Hyperparameters for GRU: (50, 32)
---------- GRU Global Model Evaluation ----------
Accuracy : 0.9917
Recall: 0.9917
Precision: 0.9921
F1 Score: 0.9917

시작 시각: 2025-07-14 14:53:06
끝 시각: 2025-07-14 15:18:11


---------------------------------------------------------------------------------------
< Transformer >
Best Hyperparameters for TF: (50, 32)
---------- TF Global Model Evaluation ----------
Accuracy : 1.0000
Recall: 1.0000
Precision: 1.0000
F1 Score: 1.0000

시작 시각: 2025-07-14 15:21:41
끝 시각: 2025-07-14 16:00:54
