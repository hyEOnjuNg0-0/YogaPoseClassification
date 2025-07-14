import tensorflow as tf
import os
import numpy as np
import pandas as pd
from keras.src.utils import to_categorical
from tensorflow import keras
import tensorflow_addons as tfa
from sklearn.utils import shuffle
from itertools import product
from sklearn.model_selection import KFold
from sklearn import metrics
from datetime import datetime

# 현재 시각 가져오기
start_time = datetime.now()

def load_data(directory_path):
    data = []
    labels = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        df = pd.read_csv(file_path)
        
        # 프레임 번호 열 삭제
        if 'frame_num' in df:
            df = df.drop(columns=['frame_num'])
        
        # feature(관절 이름), label(요가 동작 이름) 분리
        data.append(df.iloc[:, :-1])  # feature
        labels.append(df.iloc[:, -1])  # label

    return data, labels


train_directory = r'D:\yoga\yoga_lm\train'
train_input, train_labels = load_data(train_directory)
test_directory = r'D:\yoga\yoga_lm\test'
test_input, test_labels = load_data(test_directory)


# 레이블 문자열 딕셔너리
label_map = {'boat': 0,
             'bow': 1,
             'cat': 2,
             'chair': 3,
             'cobra': 4,
             'downdog': 5,
             'tree': 6,
             'warrior': 7, }

# 문자열 정수 변환
train_labels = [np.vectorize(label_map.get)(label) for label in train_labels]
test_labels = [np.vectorize(label_map.get)(label) for label in test_labels]


# 0패딩 (최대 프레임에 맞춰서)
def padding_with_zeros(data, labels, target_frames=232):
    data = data.to_numpy()  # 넘파이 배열로 변환
    num_frames, num_features = data.shape
    padded_data = np.zeros((target_frames, num_features))
    padded_labels = np.full((target_frames,), 8)

    if num_frames >= target_frames:
        return data[:target_frames, :], labels[:target_frames]
    else:
        padded_data[:num_frames, :] = data
        padded_labels[:num_frames] = labels
        return padded_data, padded_labels


# 패딩 적용
padded_train_input = []
padded_train_labels = []
for frame, label in zip(train_input, train_labels):
    padded_frame, padded_label = padding_with_zeros(frame, label)
    padded_train_input.append(padded_frame)
    padded_train_labels.append(padded_label)

padded_test_input = []
padded_test_labels = []
for frame, label in zip(test_input, test_labels):
    padded_frame, padded_label = padding_with_zeros(frame, label)
    padded_test_input.append(padded_frame)
    padded_test_labels.append(padded_label)

# 넘파이 배열로 변환
padded_train_input = np.array(padded_train_input)
padded_train_labels = np.array(padded_train_labels)
padded_test_input = np.array(padded_test_input)
padded_test_labels = np.array(padded_test_labels)

print(f'Shape Of Padded Train Input Data : {padded_train_input.shape}')
print(f'Shape Of Padded Train Labels Data : {padded_train_labels.shape}')
print(f'Shape Of Padded Test Input Data : {padded_test_input.shape}')
print(f'Shape Of Padded Test Labels Data : {padded_test_labels.shape}')


SEQUENCE_LENGTH = 33

# 시퀀스 생성 (시퀀스 길이 33)
def create_sequence_dataset(videos, video_labels, sequence_length=SEQUENCE_LENGTH, stride=33):
    sequences = []
    sequence_labels = []

    for video, label in zip(videos, video_labels):
        frame_num = len(video)
        for start in range(0, frame_num - sequence_length + 1, stride):
            end = start + sequence_length
            sequence = video[start:end]
            sequence_label = label[start]  # 시퀀스의 첫 번째 프레임 레이블
            sequences.append(sequence)
            sequence_labels.append(sequence_label)

    return np.array(sequences), np.array(sequence_labels)


sq_train_input, sq_train_labels = create_sequence_dataset(padded_train_input, padded_train_labels)
sq_test_input, sq_test_labels = create_sequence_dataset(padded_test_input, padded_test_labels)
print(f'Train input shape: {sq_train_input.shape}, Train label shape: {sq_train_labels.shape}')
print(f'Test input shape: {sq_test_input.shape}, Test label shape: {sq_test_labels.shape}')

# 데이터 섞기
sq_train_input, sq_train_labels = shuffle(sq_train_input, sq_train_labels, random_state=42)
sq_test_input, sq_test_labels = shuffle(sq_test_input, sq_test_labels, random_state=42)

# 레이블 원 핫 인코딩
encoded_sq_train_labels = to_categorical(sq_train_labels)
print(f'Shape Of Encoded Labels : {encoded_sq_train_labels.shape}')
encoded_sq_test_labels = to_categorical(sq_test_labels)
print(f'Shape Of Encoded Labels : {encoded_sq_test_labels.shape}')


input_shape = (33, 30)
num_classes = 9

# LSTM 모델
def create_lstm_model(input_shape=input_shape, num_classes=num_classes):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    model.add(keras.layers.LSTM(64, recurrent_dropout=0, return_sequences=True))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.LSTM(128, recurrent_dropout=0, return_sequences=True))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.LSTM(256, recurrent_dropout=0))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    return model


# GRU 모델
def create_gru_model(input_shape=input_shape, num_classes=num_classes):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    model.add(keras.layers.GRU(64, recurrent_dropout=0, return_sequences=True))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.GRU(128, recurrent_dropout=0, return_sequences=True))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.GRU(256, recurrent_dropout=0))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    return model


# Transformer 모델
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0):
    # Attention and Normalization
    x = keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = keras.layers.Dropout(dropout)(x)
    x1 = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x2 = tfa.layers.InstanceNormalization()(x)
    x = (0.7 * x1) + (0.3 * x2)
    res = x + inputs

    # Feed Forward Part
    x = keras.layers.Conv1D(filters=ff_dim, kernel_size=3, activation="LeakyReLU", padding='same')(res)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same')(x)
    x1 = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x2 = tfa.layers.InstanceNormalization()(x)
    x = (0.7 * x1) + (0.3 * x2)
    return x + res

# Transformer 모델 설계 부분
def build_model_1(
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0.0,
        mlp_dropout=0.0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = keras.layers.GlobalAveragePooling1D()(x)
    for dim in mlp_units:
        x = keras.layers.Dense(dim, activation="LeakyReLU")(x)
        x = keras.layers.Dropout(mlp_dropout)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)

tf_model = build_model_1(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=16,
    num_transformer_blocks=2,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.3,
)

# 레이어 별 가중치 평균 내기
def average_weights(weights_list):
    total_avg_weights = []

    num_layers = len(weights_list[0])
    for i in range(num_layers):
        weights = [weights[i] for weights in weights_list]
        avg_weights = np.mean(np.array(weights), axis=0)
        total_avg_weights.append(avg_weights)

    return total_avg_weights

# 학습할 모델 유형 ('LSTM' / 'GRU' / 'TF')
model_type = 'LSTM'  # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# NestedCV 하이퍼파라미터 그리드 설정
param_grid = {
    'epochs': [50, 100],
    'batch_size': [32, 64],
}

# 모든 하이퍼파라미터 조합 생성
param_combinations = list(product(param_grid['epochs'], param_grid['batch_size']))

# 모델 정의
model_d = {
    'LSTM_Global': create_lstm_model(input_shape, num_classes),
    'LSTM_FA_Global': create_lstm_model(input_shape, num_classes),
    'LSTM_IPA_Global': create_lstm_model(input_shape, num_classes),

    'GRU_Global': create_gru_model(input_shape, num_classes),
    'GRU_FA_Global': create_gru_model(input_shape, num_classes),
    'GRU_IPA_Global': create_gru_model(input_shape, num_classes),

    'TF_Global': tf_model,
    'TF_FA_Global': tf_model,
    'TF_IPA_Global': tf_model
}


# 모델 학습, 평가
def train_and_evaluate_model(X_train_g, y_train_g, X_valid, y_valid, epochs, batch_size):
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='min',
                                                  restore_best_weights=True)

    # 글로벌 모델
    global_model = model_d[f'{model_type}_Global']
    global_model.compile(loss='categorical_crossentropy',
                         optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                         metrics=['accuracy'])
    global_model.fit(X_train_g, y_train_g,
                     epochs=epochs,
                     batch_size=batch_size, validation_split=0.2, callbacks=[earlystopping])

    return global_model, global_model.evaluate(X_valid, y_valid, verbose=0)[1]

# 외부 및 내부 교차검증 설정
outer_cv = KFold(n_splits=5, random_state=42, shuffle=True)
inner_cv = KFold(n_splits=2, random_state=42, shuffle=True)

outer_scores = []
best_model = None
best_params = None

# 외부 교차검증
for train_idx, valid_idx in outer_cv.split(sq_train_input, encoded_sq_train_labels):

    best_inner_score, best_inner_params = 0, None
    # 내부 교차검증
    for epochs, batch_size in param_combinations:
        inner_scores = []

        for inner_train_idx, inner_valid_idx in inner_cv.split(sq_train_input[train_idx], encoded_sq_train_labels[train_idx]):
            _, score = train_and_evaluate_model(
                sq_train_input[inner_train_idx], encoded_sq_train_labels[inner_train_idx],
                sq_train_input[inner_valid_idx], encoded_sq_train_labels[inner_valid_idx],
                epochs, batch_size
            )
            inner_scores.append(score)

        mean_inner_score = np.mean(inner_scores)
        # 최적의 하이퍼파라미터 조합 업데이트
        if mean_inner_score > best_inner_score:
            best_inner_score, best_inner_params = mean_inner_score, (epochs, batch_size)

    # 최적 하이퍼파라미터로 외부 검증
    best_epochs, best_batch_size = best_inner_params
    trained_model, outer_score = train_and_evaluate_model(
        sq_train_input[train_idx], encoded_sq_train_labels[train_idx],
        sq_train_input[valid_idx], encoded_sq_train_labels[valid_idx],
        best_epochs, best_batch_size
    )
    outer_scores.append(outer_score)

    # 가장 높은 성능을 가진 모델 저장
    if outer_score == max(outer_scores):
        best_model, best_params = trained_model, best_inner_params

print('\tOuter loop : ')
print(f'Mean Acc : {np.mean(outer_scores):.4f}, Std Acc : {np.std(outer_scores):.4f}')

# 글로벌 모델 평가
model_evaluation = best_model.evaluate(sq_test_input, encoded_sq_test_labels)
model_loss, model_acc = model_evaluation

pred_labels = np.argmax(best_model.predict(sq_test_input), axis=1)
true_labels = np.argmax(encoded_sq_test_labels, axis=1)

# 모델 저장 디렉토리
save_directory = r'D:\yoga\Deeplearningmodels'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

model_file_name = os.path.join(save_directory, f'{model_type}_NCV__Loss_{model_loss:.4f}__Acc_{model_acc:.4f}.h5')
best_model.save(model_file_name)
print(f'Saved {model_type} model')


recall = metrics.recall_score(true_labels, pred_labels, average='weighted')
precision = metrics.precision_score(true_labels, pred_labels, average='weighted')
f1 = metrics.f1_score(true_labels, pred_labels, average='weighted')

# 최적의 하이퍼파라미터 출력
print(f"Best Hyperparameters for {model_type}: {best_params}")
print(f'---------- {model_type} Global Model Evaluation ----------')
print(f'Accuracy : {model_acc:.4f}')
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")

# 현재 시각 가져오기
end_time = datetime.now()
# 현재 시각 출력
print("\n시작 시각:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
print("끝 시각:", end_time.strftime("%Y-%m-%d %H:%M:%S"))








