# -*- coding: utf-8 -*-

"""
Author: Pastoral, Lorenzo Troy
Date Created: 15/09/2021
Hand Tracker
Description: A simple program that tracks the user's hand.
Module Version - Can be used on other projects. This program utilizes OOP to make it more modular and reusable.
"""

# Main Modules
import cv2
import mediapipe as mp
import time

# Main Class


class handDetector():

    # Initializes itself
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,  trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    # Looks for hand via mediapipe module
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:  # Draws Points Only
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS)  # Draws Conections
        return img

    # Draws and tracks hand
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:  # Draws Points Only
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    # Change Color and Attributes of Hand Model Tracking
                    cv2.circle(img, (cx, cy), 5, (255, 255, 0), cv2.FILLED)

        return lmList

# Run Main Code / Template Code


def main():

    # Previous and Current time for FPS
    pTime = 0
    cTime = 0

    # Capture webcam | choose webcam (1-3). Depends on how many camera inputs you have
    cap = cv2.VideoCapture(1)
    detector = handDetector()  # Creates instance of object

    # Runs Actual Program
    while True:
        success, img = cap.read()
        img = detector.findHands(img)  # Use Method Here
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            # Print values at index (each part of the finger has a unique ID)
            print(lmList[4])
            print("Thumb")
        # Display FPS Counter
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 25, 255), 3)  # Change FPS Attributes

        cv2.imshow("Image", img)
        cv2.waitKey(1)


# Run Window
if __name__ == "__main__":
    main()
