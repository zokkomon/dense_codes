import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import numpy as np
# from pynput.keyboard import Controller

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = HandDetector(detectionCon=0.8)
# keyboard = Controller()

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.text = text
        self.size = size
        
def drawButton(frame, buttonList):
    imgNew = np.zeros_like(frame, np.uint8)
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(imgNew, (button.pos[0], button.pos[1], button.size[0], button.size[1]), 20, rt=0)
        cv2.rectangle(imgNew, button.pos, (x+w, y+h), (255, 255, 0), cv2.FILLED)
        cv2.putText(imgNew, button.text, (x+20, y+65), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4)
        
    # Generate output by blending image with shapes image, using the shapes
    # images also as mask to limit the blending to those parts
    out = frame.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(frame, alpha, imgNew, 1 - alpha, 0)[mask]
    return out

keys =  [[ 'q', 'w', 'e',  'r', 't', 'y', 'u', 'i', 'o', 'p' ], 
          [ 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l' ], 
          [ 'z', 'x', 'c', 'v', 'b', 'n','m','']]  
finalText = ""
    
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100*j+50, 100*i+50], key))              

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = drawButton(frame, buttonList) 
    # frame = cv2.flip(frame,1)
    hands, frame = detector.findHands(frame)
    
    if hands:
        lmList = hands[0]["lmList"]
        
        if lmList:
            for button in buttonList:
                x, y = button.pos
                w, h = button.size
                
                if x < lmList[8][0] < x+w and y < lmList[8][1] < y+h:
                    cv2.rectangle(frame, button.pos, (x+w, y+h), (0, 0, 255), cv2.FILLED)
                    cv2.putText(frame, button.text, (x+20, y+65), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4)
                    
                    length, _, _ = detector.findDistance(lmList[8][0:2], lmList[12][0:2], frame)
                    # print(length)
                    
                    if length<60:
                        # keyboard.press(button.text)
                        cv2.rectangle(frame, button.pos, (x+w, y+h), (0, 255, 0), cv2.FILLED)
                        cv2.putText(frame, button.text, (x+20, y+65), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4)
                        finalText += button.text
                        sleep(0.10)
                        
    cv2.rectangle(frame, (50,350), (700,450), (236, 135, 0), cv2.FILLED)
    cv2.putText(frame, finalText, (60, 430), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4)    
                 
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()