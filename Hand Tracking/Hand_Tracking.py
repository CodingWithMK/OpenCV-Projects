import cv2
import mediapipe as mp

camIndex = 0
capture = cv2.VideoCapture(camIndex)
npHands = mp.solutions.hands
Hands = npHands.Hands()
npDraw = mp.solutions.drawing_utils

while (capture.isOpened()):
    success_, img = capture.read()
    cvtImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = Hands.process(cvtImg)

    if result.multi_hand_landmarks:
        for img_in_frame in result.multi_hand_landmarks:
            npDraw.draw_landmarks(img, img_in_frame, npHands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) == 113: # Q = 113
        break



