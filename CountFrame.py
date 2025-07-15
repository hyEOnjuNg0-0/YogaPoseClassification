##################################################
########### 비디오 최대/최소 프레임 확인   ###########
#################################################

import os
import cv2

# 요가 비디오가 저장된 최상위 디렉토리
base_path = r"D:\yoga\yoga_video"

min_frames = float('inf')
max_frames = float('-inf')
min_file = ""
max_file = ""

# 모든 하위 폴더와 파일을 순회
for pose_name in os.listdir(base_path):
    pose_path = os.path.join(base_path, pose_name)

    # 폴더인 경우에만 처리
    if os.path.isdir(pose_path):
        for file_name in os.listdir(pose_path):
            file_path = os.path.join(pose_path, file_name)

            # 확장자가 '.mp4'인 경우만 처리
            if file_path.lower().endswith('.mp4'):
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    print(f"Cannot open video: {file_path}")
                    continue

                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                # 최소 / 최대 프레임 업데이트
                if frame_count < min_frames:
                    min_frames = frame_count
                    min_file = file_path
                if frame_count > max_frames:
                    max_frames = frame_count
                    max_file = file_path

print("최소 프레임 :")
print(f"{min_file} - {min_frames} frames")

print("\n최대 프레임 :")
print(f"{max_file} - {max_frames} frames")
