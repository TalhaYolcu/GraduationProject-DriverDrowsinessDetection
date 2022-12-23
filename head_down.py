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


right_nose_index =49
left_nose_index= 279
up_nose_index = 195
up_nose_index_2 = 5




WAIT_TIME=5.0
D_TIME=0.0


mixer.init()
mixer.music.load("alarm.mp3")



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
            for i in range(0, 438):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                if i==left_nose_index or i==right_nose_index or i==up_nose_index or i==up_nose_index_2:
                    cv2.circle(image, (x, y), 2, (100, 100, 0), -1)


        right_nose_point = facial_landmarks.landmark[right_nose_index]
        left_nose_point = facial_landmarks.landmark[left_nose_index]
        up_nose_point_y = (facial_landmarks.landmark[up_nose_index].y+ 2*facial_landmarks.landmark[up_nose_index_2].y) / 3



        middle_point_y=(right_nose_point.y + left_nose_point.y) / 2

        if middle_point_y <= up_nose_point_y :
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
