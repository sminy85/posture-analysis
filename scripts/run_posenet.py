import cv2
import sys
import os

# 'models' 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from models.posenet_model.model import load_posenet_model
from models.posenet_model.model import get_keypoints
from models.posture_feedback import analyze_posture

# PoseNet 모델 로드
model = load_posenet_model()

# 비디오 캡처
cap = cv2.VideoCapture(0)  # 0번은 기본 웹캠

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 포즈 추정
    keypoints = get_keypoints(frame, model)

    # 자세 분석 및 피드백 제공
    feedback = analyze_posture(keypoints)
    cv2.putText(frame, feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 결과 화면 표시
    cv2.imshow('PoseNet - Exercise Feedback', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
