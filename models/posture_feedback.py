def analyze_posture(keypoints):
    keypoints = keypoints[0]  # 첫 번째 배치 선택

    try:
        knee_y, knee_x, knee_confidence = keypoints[8]  # 무릎 (index 8)
        hip_y, hip_x, hip_confidence = keypoints[9]    # 엉덩이 (index 9)
    except IndexError:
        return "Error in keypoints extraction"

    # print(f"hip_y: {hip_y}, knee_y: {knee_y}, hip_confidence: {hip_confidence}, knee_confidence: {knee_confidence}")

    # confidence가 낮은 경우에도 y값을 기준으로 피드백을 제공
    #if knee_confidence < 0.2 or hip_confidence < 0.2:
    #    print("Low confidence detected, but proceeding based on y position.")

    # 엉덩이가 무릎보다 높으면 더 앉으라고 피드백
    if hip_y > knee_y + 0.05:  # 기준을 조금 유연하게 설정 (ex: 5%)
        return "Your hips are too high, try to squat deeper!"
    # 엉덩이가 무릎과 같거나 낮으면 잘하고 있다고 피드백
    elif hip_y <= knee_y + 0.05:
        return "Good squat position!"
    else:
        return "Check your posture, keep your hips aligned with your knees."
