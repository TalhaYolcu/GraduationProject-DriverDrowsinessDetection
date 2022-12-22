


import cv2
import mediapipe as mp
import time

import numpy

from functions import get_points
from pygame import mixer
from functions import calculate_slope
from functions import calculate_angle





#init face mesh
mp_face_mesh=mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh()

#find video source
cap=cv2.VideoCapture(0)

height=-1
width=-1


#set angle
angle = 45

WAIT_TIME=5.0
D_TIME=0


mixer.init()
mixer.music.load("alarm.mp3")

#sag yanak
right_cheek_index=93
#sol yanak
left_cheek_index=323
#cene
chin_index=152


t1 = time.time()

isDrowsy=False
isDone=False

while True:

    #read image
    ret,image = cap.read()

    if ret is False :
        break

    #take image's width and height
    if(height==-1 | width==-1):
        height,width,_ = image.shape


    rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    #process image

    result = face_mesh.process(rgb_image)

    if result.multi_face_landmarks:
        for facial_landmarks in result.multi_face_landmarks:

            for i in range (0,438):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x*width)
                y=int(pt1.y*height)

                cv2.circle(image,(x,y),2,(100,100,0),-1)



        right_cheek_point=facial_landmarks.landmark[right_cheek_index]
        left_cheek_point=facial_landmarks.landmark[left_cheek_index]
        chin_point=facial_landmarks.landmark[chin_index]

        slope_right,slope_left=calculate_slope(right_cheek_point,left_cheek_point,chin_point)

        angle_right,angle_left=calculate_angle(slope_right,slope_left)

        print(angle_right,angle_left)

        if(-0.5<angle_left<0) :
            t2=time.time()
            time_t=t2-t1
            D_TIME=D_TIME+time_t
            t1=t2

            if D_TIME>=WAIT_TIME:
                if(isDone==False):
                    mixer.music.play()

                isDrowsy=True
                isDone=True

        elif(0<angle_right<0.6) :
            t2=time.time()
            time_t=t2-t1
            D_TIME=D_TIME+time_t
            t1=t2

            if D_TIME>=WAIT_TIME:
                if(isDone==False):
                    mixer.music.play()

                isDrowsy=True
                isDone=True


        cv2.imshow("Image",image)

        cv2.waitKey(1)


    else :
        D_TIME=0
        t1=time.time()

