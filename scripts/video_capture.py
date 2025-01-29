import cv2
from models.posenet_model.model import load_posenet_model, get_keypoints

# PoseNet 모델 로드
model = load_posenet_model()

# 비디오 캡처
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 포즈 추정
    keypoints = get_keypoints(frame, model)

    # 관절 표시
    for i in range(keypoints.shape[0]):
        y, x, confidence = keypoints[i]
        if confidence > 0.5:  # Confidence가 50% 이상일 때만 그리기
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # 결과 화면 표시
    cv2.imshow('PoseNet - Exercise Feedback', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
