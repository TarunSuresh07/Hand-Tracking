import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True :
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    result = hands.process(imgRGB)
    # print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks :
        for handLand in result.multi_hand_landmarks :
            for id,lm in enumerate(handLand.landmark):
                print(id,lm)
            mpDraw.draw_landmarks(img, handLand, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_DUPLEX, 3, (255,0,0), 3)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)