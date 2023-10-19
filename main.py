import math
import time
import numpy as np
import cv2 #pip install opencv-python ||| pip3 install opencv-contrib-python==4.4.0.46

from escpos.printer import Serial

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from deepface import DeepFace #pip install deepface

cam = cv2.VideoCapture(1)

printer = Serial(devfile='/dev/serial0',baudrate=19200,bytesize=8,parity='N',stopbits=1.00,dsrdtr=True)
printer.set(density=10)
cameraMode = False
TIMER = 5

startScreen = cv2.imread("noFace.png")

if cam.isOpened():
    while True:
        ret, img = cam.read()
        if cameraMode and ret:
            prev = time.time() 
            while TIMER > 0:
                ret, img = cam.read()
                # cv2.putText(img, str(TIMER), (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 255, 255), 4, cv2.LINE_AA) 
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font_size = 350
                font = ImageFont.truetype("HalloweenFont.ttf", font_size)
                if TIMER>1:
                    draw.text((231, 50), str(TIMER), font=font,fill=(255,0,0,255))
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                cv2.imshow('webcam',img)
                # print(str(TIMER))
                cur = time.time() #current time
                if cur-prev >= 1: 
                    prev = cur
                    TIMER = TIMER-1
                key = cv2.waitKey(5) & 0xFF
                if ord('q') == key:
                    break
            else:
                ret, img = cam.read()
                predictions = DeepFace.analyze(img,actions=['emotion'])
                fearPoint = predictions[0]["emotion"]["fear"]
                print("FEAR:" + str(fearPoint))

                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                font_size = 65
                font = ImageFont.truetype("HalloweenFont.ttf", font_size)
                text = "FEAR LEVEL"
                draw.text((144, 308), str(text), font=font,fill=(255,0,0,255))
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                #print(30+math.floor(int(fearPoint)*580/100))
                # ft.putText(img=img,text='TEST',org=(15, 70),fontHeight=60,color=(255,  255, 255),thickness=-1,line_type=cv2.LINE_AA,bottomLeftOrigin=True)
                cv2.rectangle(img,(30,400),(610,450),(255,255,255), 5)
                cv2.rectangle(img,(30,400),(30+math.floor(int(fearPoint)*580/100),450),(255,255,255), -1)
                
                if fearPoint>10:
                    cv2.imwrite('scared.jpg', img) 
                    basewidth = 384
                    imgCrop = Image.open('scared.jpg')
                    wpercent = (basewidth/float(imgCrop.size[0]))
                    hsize = int((float(imgCrop.size[1])*float(wpercent)))
                    imgCrop = imgCrop.resize((basewidth,hsize), Image.Resampling.LANCZOS)
                    imgCrop = imgCrop.save("cropScared.jpg")
                    cv2.waitKey(500)
                    printer.image(cropScared.jpg)
                    printer.text("\n\n\n\n")               
                cv2.imshow('webcam',img)
                cv2.waitKey(5000)



                cameraMode = False
                TIMER = 5
        else:
            cv2.imshow('webcam',startScreen) 
        
        key = cv2.waitKey(5) & 0xFF
        if ord('t') == key:
            cameraMode = True
        if ord('q') == key:
            break
           
cam.release()