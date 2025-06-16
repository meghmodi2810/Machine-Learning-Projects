from Modules.HandDetectionModule import HandDetectionModule
from Modules.PoseDetectionModule import PoseDetectionModule
from Modules.InteractionAnalyzer import InteractionAnalyzer
from LearningModules.MLInteractionAnalyzer import MLInteractionAnalyzer
import cv2
import numpy as np
import time

# Initialize
config = {
    'alert_threshold': 50,
    'model_path': 'touch_model.joblib',
    'max_hands': 2,
    'hand_detection_confidence': 0.7,
    'hand_tracking_confidence': 0.5,
    'pose_detection_confidence': 0.7,
    'pose_tracking_confidence': 0.5,
    'privacy_blur': True
}

hand_detector = HandDetectionModule(
    maxHands=config['max_hands'],
    detectionCon=config['hand_detection_confidence'],
    trackCon=config['hand_tracking_confidence']
)

pose_detector = PoseDetectionModule(
    detectionCon=config['pose_detection_confidence'],
    trackCon=config['pose_tracking_confidence']
)

analyzer = InteractionAnalyzer(
    safe_threshold=config['alert_threshold']
)

ml_analyzer = MLInteractionAnalyzer(
    model_path=config['model_path']
)

last_alert_time = 0

def apply_privacy_blur(img, body_points):
    for zone in ['groin', 'buttocks']:
        if zone in body_points:
            x, y = body_points[zone]
            img[y - 50:y + 50, x - 30:x + 30] = cv2.GaussianBlur(
                img[y - 50:y + 50, x - 30:x + 30], (51, 51), 0
            )
    return img

def get_body_points(pose_landmarks, img):
    body_points = {}
    if not pose_landmarks or len(pose_landmarks) < 25:
        return body_points
    try:
        l_shoulder = pose_landmarks[11][1:]
        r_shoulder = pose_landmarks[12][1:]
        chest = ((l_shoulder[0] + r_shoulder[0]) // 2, (l_shoulder[1] + r_shoulder[1]) // 2)
        body_points['chest'] = chest

        l_hip = pose_landmarks[23][1:]
        r_hip = pose_landmarks[24][1:]
        waist = ((l_hip[0] + r_hip[0]) // 2, (l_hip[1] + r_hip[1]) // 2)
        body_points['waist'] = waist

        groin = ((l_hip[0] + r_hip[0]) // 2, (l_hip[1] + r_hip[1]) // 2 + int(0.05 * img.shape[0]))
        body_points['groin'] = groin

        buttocks = (groin[0], groin[1] + int(0.15 * img.shape[0]))
        body_points['buttocks'] = buttocks

        nose = pose_landmarks[0][1:]
        mouth = (nose[0], nose[1] + int(0.05 * img.shape[0]))
        body_points['mouth'] = mouth
    except IndexError:
        pass
    return body_points

def process_frame(img):
    global last_alert_time

    img, hands_data = hand_detector.multiHandFinder(img, draw=True)
    hands_landmarks = hand_detector.multiHandPositionFinder(img, draw=True)
    img = pose_detector.poseFinder(img, draw=True)
    pose_landmarks = pose_detector.positionFinder(img, draw=True)

    body_points = get_body_points(pose_landmarks, img)

    if config['privacy_blur']:
        img = apply_privacy_blur(img, body_points)

    if hands_landmarks and body_points:
        for hand in hands_landmarks:
            if not hand:
                continue

            hand_fingertips = [(x, y) for _, id, x, y in hand if id in [8, 12, 16, 20]]

            if hand_fingertips:
                zone, distance = analyzer.analyze(hand_fingertips, body_points)

                if zone != "none":
                    current_time = int(time.time() * 1000)
                    if current_time - last_alert_time > 500:
                        print(f"[ALERT] Touch near {zone.upper()} (distance={distance:.2f})")
                        last_alert_time = current_time
                        cv2.putText(
                            img, f"ALERT: Touch near {zone.upper()}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                        )

    return img
