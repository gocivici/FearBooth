import math
import numpy as np
import cv2 #pip install opencv-python ||| pip3 install opencv-contrib-python==4.4.0.46

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from deepface import DeepFace #pip install deepface

readings = np.array([])
max_samples = 10
# img = cv2.imread("test.jpg")
cam = cv2.VideoCapture(1)
# cam.set(cv2.cv.CV_CAP_PROP_FPS, 10)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
noFace = cv2.imread("noFace.png")

# ft = cv2.freetype.createFreeType2()
# ft.loadFontData(fontFileName='HalloweenFont.ttf',id=0)
# overlay = cv2.imread('feartext.png')
#custom resolution for CRT TV
# cam.set(3,768)
# cam.set(4,576)

#640 x 480

if cam.isOpened():
    while True:
        ret, img = cam.read()
        if ret:
            img = cv2.resize(img, (80,60), interpolation = cv2.INTER_AREA)
            print(img.shape)
            try:
                predictions = DeepFace.analyze(img,actions=['emotion'])
                fearPoint = predictions[0]["emotion"]["fear"]

                readings = np.append(readings, fearPoint)
                avg = np.mean(readings)
                if len(readings) == max_samples:
                    readings = np.delete(readings, 0)
                print("FEAR:" + str(fearPoint))
                print("AVG" + str(avg))
                
                # for x in range(6):
                #     avgNumber = avgNumber + fearPoint

                # print(avgNumber)
                # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
                # img = Image.fromarray(img)
                # draw = ImageDraw.Draw(img)
                # font_size = 65
                # font = ImageFont.truetype("HalloweenFont.ttf", font_size)
                # text = "FEAR LEVEL"
                # draw.text((144, 308), str(text), font=font,fill=(255,0,0,255))
                # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                #print(30+math.floor(int(fearPoint)*580/100))
                # ft.putText(img=img,text='TEST',org=(15, 70),fontHeight=60,color=(255,  255, 255),thickness=-1,line_type=cv2.LINE_AA,bottomLeftOrigin=True)
                # cv2.rectangle(img,(30,400),(610,450),(255,255,255), 5)
                # cv2.rectangle(img,(30,400),(30+math.floor(int(avg)*580/100),450),(255,255,255), -1)
                cv2.imshow('webcam',img)
            except Exception as e: 
                print(e)

                cv2.imshow('webcam',noFace)
        key = cv2.waitKey(5) & 0xFF
        if ord('q') == key:
            break
cam.release()