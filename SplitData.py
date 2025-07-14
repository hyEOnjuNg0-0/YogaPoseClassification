#############################################################################
########### 전체 csv 파일 660개 랜덤하게 train(80%), test(20%)로 분리 ###########
#############################################################################

import pandas as pd
import os
import numpy as np

###################### 전체 데이터 train / test ######################
data_path = os.listdir('D:\yoga\yoga_lm')

all_files = []
for file_name in data_path:
    all_files.append(file_name)

raw_df = pd.DataFrame(data=all_files, columns=['file_name'])
print('---------------------------- RAW DATA LIST ----------------------------')
print(raw_df.head())
print(raw_df.tail())
raw_df.to_csv(r'D:\yoga\yoga_lm\raw_list.csv')

# 파일을 랜덤하게 섞기
np.random.seed(42)
np.random.shuffle(all_files)

# 훈련 데이터, 테스트 데이터 8:2로 나누기
split_index = int(0.8 * len(all_files))
train_files = all_files[:split_index]
test_files = all_files[split_index:]

train_df = pd.DataFrame(data=train_files, columns=['file_name'])
print('---------------------------- TRAIN DATA LIST ----------------------------')
print(train_df.head())
print(train_df.tail())
train_df.to_csv('train_list.csv')

test_df = pd.DataFrame(data=test_files, columns=['file_name'])
print('---------------------------- TEST DATA LIST ----------------------------')
print(test_df.head())
print(test_df.tail())
test_df.to_csv('test_list.csv')

# 파일 이동
def move_files(file_list, origin_directory, destination_directory):
    for file in file_list:
        try:
            file_name = os.path.basename(file)
            destination_path = os.path.join(destination_directory, file_name)
            file_path = os.path.join(origin_directory, file)
            os.rename(file_path, destination_path)
            print(f"파일 이동됨: {file} -> {destination_path}")
        except Exception as e:
            print(f"파일 이동 중 오류 발생: {e}")

origin_directory = 'D:\yoga\yoga_lm'
train_directory = r'D:\yoga\yoga_lm\train'
test_directory = r'D:\yoga\yoga_lm\test'

os.makedirs(train_directory, exist_ok=True)
os.makedirs(test_directory, exist_ok=True)

move_files(train_files, origin_directory, train_directory)
move_files(test_files, origin_directory, test_directory)




# ###################### train 데이터 train / val ######################
# data_path = os.listdir('/mnt/storage01/yoga_pose_lm/train')
#
# all_train_files = []
# for file_name in data_path:
#     all_train_files.append(file_name)
#
# train_df = pd.DataFrame(data=all_train_files, columns=['file_name'])
# print('---------------------------- RAW TRAIN DATA LIST ----------------------------')
# print(train_df.head())
# print(train_df.tail())
# # raw_df.to_csv('raw_list.csv')
#
# # 파일을 랜덤하게 섞기
# np.random.seed(42)
# np.random.shuffle(all_train_files)
#
# # 훈련 데이터, 검증 데이터 8:2로 나누기
# split_index = int(0.8 * len(all_train_files))
# train_files = all_train_files[:split_index]
# val_files = all_train_files[split_index:]
#
# train_df = pd.DataFrame(data=train_files, columns=['file_name'])
# print('---------------------------- TRAIN DATA LIST ----------------------------')
# print(train_df.head())
# print(train_df.tail())
# train_df.to_csv('train_list.csv')
#
# val_df = pd.DataFrame(data=val_files, columns=['file_name'])
# print('---------------------------- VAL DATA LIST ----------------------------')
# print(val_df.head())
# print(val_df.tail())
# val_df.to_csv('val_list.csv')
#
# # 파일 이동
# def move_files(file_list, origin_directory, destination_directory):
#     for file in file_list:
#         try:
#             file_name = os.path.basename(file)
#             destination_path = os.path.join(destination_directory, file_name)
#             file_path = os.path.join(origin_directory, file)
#             os.rename(file_path, destination_path)
#             print(f"파일 이동됨: {file} -> {destination_path}")
#         except Exception as e:
#             print(f"파일 이동 중 오류 발생: {e}")
#
# train_directory = '/mnt/storage01/yoga_pose_lm/train'
# val_directory = '/mnt/storage01/yoga_pose_lm/val'
#
# os.makedirs(train_directory, exist_ok=True)
# os.makedirs(val_directory, exist_ok=True)
#
# move_files(val_files, train_directory, val_directory)