import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mphands = mp.solutions.hands

cap = cv2.VideoCapture(0)
hands = mphands.Hands()

while True:
    ret, image = cap.read()
    if not ret:
        continue

    image = cv2.flip(image, 1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                hand_landmarks,
                mphands.HAND_CONNECTIONS)
            
        # Display the image
        cv2.imshow('Handtracker', image_bgr)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()