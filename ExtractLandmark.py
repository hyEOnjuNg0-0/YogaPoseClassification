import cv2
import mediapipe as mp
import pandas as pd
import os

class poseDetector:
    # mediapipe에서 제공하는 기본값 사용
    def __init__(self, mode=False, mComp=1, smoothLM=True, enableSeg=False,
                 smoothSeg=True, detectionConf=0.5, trackingConf=0.5):
        self.mode = mode
        self.mComp = mComp
        self.smoothLM = smoothLM
        self.enableSeg = enableSeg
        self.smoothSeg = smoothSeg
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.mComp, self.smoothLM, self.enableSeg, self.smoothSeg,
                                      self.detectionConf, self.trackingConf)
        self.mp_draw = mp.solutions.drawing_utils

    def crop_center_square(self, img):
        y, x = img.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return img[start_y: start_y + min_dim, start_x: start_x + min_dim]

    # 랜드마크, 선 그리기 (선택 가능, 기본 True)
    def findPose(self, img, draw=True):
        # 모든 비디오 프레임 크기 동일하게
        img = self.crop_center_square(img)
        img = cv2.resize(img, (224, 224))

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return img

    # 필요 관절 위치 얻기 (정수로)
    def getPosition(self, img):
        lm_list = []

        # 랜드마크 딕셔너리
        BODY_PARTS = {
            0: "Nose",
            1: "LEyeInner",
            2: "LEye",
            3: "LEyeOuter",
            4: "REyeInner",
            5: "REye",
            6: "REyeOuter",
            7: "LEar",
            8: "REar",
            9: "MouthL",
            10: "MouthR",
            11: "LShoulder",
            12: "RShoulder",
            13: "LElbow",
            14: "RElbow",
            15: "LWrist",
            16: "RWrist",
            17: "LPinky",
            18: "RPinky",
            19: "LIndex",
            20: "RIndex",
            21: "LThumb",
            22: "RThumb",
            23: "LHip",
            24: "RHip",
            25: "LKnee",
            26: "RKnee",
            27: "LAnkle",
            28: "RAnkle",
            29: "LHeel",
            30: "RHeel",
            31: "LFootIndex",
            32: "RFootIndex",
        }

        desired_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30]

        if self.results.pose_landmarks:
            for id in desired_landmarks:
                lm = self.results.pose_landmarks.landmark[id]
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                body_part_name = BODY_PARTS.get(id, "Unknown Part")
                lm_list.append((body_part_name, cx, cy))

        return lm_list

def main():
    detector = poseDetector()

    data_path = os.listdir(r'D:\yoga\yoga_video')
    print('------------- LABEL_LIST -------------')
    print(data_path)
    print('\n')

    files = []
    for label in data_path:
        num = 1
        # data_path의 모든 파일명 all_labels 리스트에 저장
        all_labels = os.listdir(r'D:\yoga\yoga_video' + '/' + label)
        # 파일 경로명과 카테고리를 튜플로 저장
        for file_name in all_labels:
            files.append((str(r'D:\yoga\yoga_video' + '/' + label) + '/' + file_name, label))
            video_path = str(r'D:\yoga\yoga_video' + '/' + label) + '/' + file_name
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error opening video file: {video_path}")
                continue

            all_frames_data = []
            frame_num = 0
            while True:
                success, img = cap.read()
                if not success:
                    break

                img = detector.findPose(img, False)

                lm_list = detector.getPosition(img)

                # lm_list가 빈 경우 그 프레임 건너뜀
                if not lm_list:
                    print(f"Skipping frame {frame_num} due to empty lm_list")
                    continue

                frame_data = {'frame_num': frame_num}
                for body_part, x, y in lm_list:
                    frame_data[f'{body_part}_x'] = x
                    frame_data[f'{body_part}_y'] = y
                    
                # pose_label 열 추가
                frame_data['Pose_label'] = label

                # # level 열 추가
                # if '초급' in file_name:
                #     frame_data['Level'] = 'Beginner'
                # elif '고급' in file_name:
                #     frame_data['Level'] = 'Advanced'
                # else:
                #     frame_data['Level'] = 'Unknown'

                all_frames_data.append(frame_data)
                frame_num += 1

                # cv2.imshow("image", img)
                # cv2.waitKey(1)

            lm_df = pd.DataFrame(all_frames_data)

            # csv 파일로 저장
            lm_filename = f'D:/yoga/yoga_lm/{label}_{num}.csv'
            lm_df.to_csv(lm_filename, index=False)
            print(f'Saved {lm_filename} --------------------------')

            num += 1
            cap.release()

    # 비디오 데이터 프레임
    print('------------- VIDEO_DATA_LIST -------------')
    video_df = pd.DataFrame(data=files, columns=['video_path', 'label'])
    print(video_df.head())
    print(video_df.tail())
    video_df.to_csv('yoga_video_list.csv')

if __name__ == "__main__":
    main()

