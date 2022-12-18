import cv2
import mediapipe as mp

camIndex = 0
capture = cv2.VideoCapture(camIndex)
npHands = mp.solutions.hands
Hands = npHands.Hands()
npDraw = mp.solutions.drawing_utils
FingersCoordinates = [(8, 6), (12, 10), (16, 14), (20, 18)]
ThumbCoordinates = (4, 3)

while (capture.isOpened()):
    upcount = 0
    success_, img = capture.read()
    cvtImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = Hands.process(cvtImg)
    HandNum = 0
    lmlist = []

    if result.multi_hand_landmarks:
        for id, lm in enumerate(result.multi_hand_landmarks[HandNum].landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmlist.append((cx, cy))

        for img_in_frame in result.multi_hand_landmarks:
            npDraw.draw_landmarks(img, img_in_frame, npHands.HAND_CONNECTIONS)
        
        for point in lmlist:
            cv2.circle(img, point, 5, (0, 255, 0), cv2.FILLED)

        for coordinates in FingersCoordinates:
            if lmlist[coordinates[0]][1] < lmlist[coordinates[1]][1]:
                upcount += 1
        
        if lmlist[ThumbCoordinates[0]][0] > lmlist[ThumbCoordinates[1]][0]:
            upcount += 1
        
        cv2.putText(img, str(upcount), (50, 150), cv2.FONT_HERSHEY_PLAIN, 12, (0, 0, 255), 12)


    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) == 113: # Q = 113
        break



