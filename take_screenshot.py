import numpy as np
import cv2
import imutils
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(1)

finger_tips = [8, 12, 16, 20]

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = [lm for lm in hand_landmark.landmark]
            finger_fold_status = []

            for tip in finger_tips:
                x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)
                cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)

                if lm_list[tip].x < lm_list[tip - 3].x:
                    cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            print(finger_fold_status)

            if all(finger_fold_status):
                screenshot = pyautogui.screenshot()
                screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                cv2.imwrite('in_memory_to_disk.png', screenshot)
                cv2.imshow('Screenshot', imutils.resize(screenshot, width=600))
            pyautogui.screenshot('straight_to_disk.png')
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2))
            image = cv2.imread('straight_to_disk.png')
            cv2.imshow('Screenshot', imutils.resize(image, width=600))

    cv2.imshow("hand tracking", img)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
