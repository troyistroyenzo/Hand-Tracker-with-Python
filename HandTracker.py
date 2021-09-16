# -*- coding: utf-8 -*-

"""
Author: Pastoral, Lorenzo Troy
Date Created: 15/09/2021
Hand Tracker
Description: A simple program that tracks the user's hand.
Plain program - No OOP utilized
"""

# Main Modules

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)  # Id Of webcam

mpHands = mp.solutions.hands
hands = mpHands.Hands(5)
mpDraw = mp.solutions.drawing_utils


pTime = 0  # previous time
cTime = 0  # current ime


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(result.multi_hand_landmark) - detects hand

    if results.multi_hand_landmarks:  # points only
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 30, (255, 0, 255), cv2.FILLED)

                mpDraw.draw_landmarks(
                    img, handLms, mpHands.HAND_CONNECTIONS)  # draw connections

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 1), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
