import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

import tensorflow_hub as hub

def load_posenet_model():
    # TensorFlow Hub에서 모델 로드
    model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
    posenet_model = hub.load(model_url)

    # 모델 확인
    print(f"Model loaded: {posenet_model}")

    # signature = posenet_model.signatures['serving_default']

    # 서명의 입력 텐서 형태 확인
    # print("!!!!!!!!!!!", signature.structured_input_signature)

    return posenet_model


# 포즈 추정 함수
def get_keypoints(frame, model):

    # 이미지를 RGB로 변환 (OpenCV는 BGR로 이미지를 읽기 때문에 RGB로 변환해야 합니다)
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 모델의 입력 크기에 맞게 리사이즈 (192x192)
    input_image = cv2.resize(input_image, (192, 192))  # 모델이 요구하는 크기에 맞게 수정

    # 이미지를 Tensor로 변환하고, 배치 차원을 추가
    input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)

    # 이미지 정규화 (0-255 -> 0-1 범위)
    input_image = input_image / 255.0

    # 배치 차원 추가 (모델은 [batch_size, height, width, channels] 형식의 입력을 요구)
    input_image = input_image[tf.newaxis, ...]

    # 데이터 타입을 int32로 변경
    input_image = tf.cast(input_image, dtype=tf.int32)

    # 모델의 'serving_default' signature로 예측 수행
    outputs = model.signatures['serving_default'](input_image)

    # 출력에서 keypoints 가져오기
    keypoints = outputs['output_0']

    # print("Extracted Keypoints: ", keypoints)  # 디버깅용 출력

    return keypoints[0]  # 첫 번째 프레임의 관절 좌표 반환