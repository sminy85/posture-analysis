import cv2
import sys
import os
# 'models' 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from models.posenet_model.model import load_posenet_model
from models.posenet_model.model import get_keypoints
from models.posture_feedback import analyze_posture



def process_video(video_path):
    model = load_posenet_model()

    cap = cv2.VideoCapture(video_path)  # 동영상 파일 열기

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished processing the video.")
            break

        frame_count += 1
        # print(f"Processing frame {frame_count}")

        # 포즈 추정
        keypoints = get_keypoints(frame, model)

        # 자세 분석 및 피드백 제공
        feedback = analyze_posture(keypoints)

        # 피드백을 로그로 출력
        print(f"Frame {frame_count} Feedback: {feedback}")

    cap.release()

# 동영상 파일 경로 지정
video_path = 'videos/squat_video.mov'  # videos 폴더에 있는 동영상 파일 경로

# 동영상 파일을 읽고 피드백을 로그로 출력
process_video(video_path)