import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title(' Attendance! ')
Face=[]
import cv2 

hello=""


# For ClockIn on button click
def crop(imgName):
    # Read the input image
    print(imgName)
    # img = cv2.imread("V/"+imgName)
    img=imgName
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w+50, y+h+50), (0, 0, 255), 2)
        faces = img[y:y + h, x:x + w]
        # cv2.imshow('face',faces)
        Face.append(faces)
        # Converting numpy array to Pil image
        image = Image.fromarray(np.uint8(faces)).convert('RGB')
    image=image.resize((256,256))
# image=np.array(image)
    image = np.array(image,dtype='float32')/255
# image=image.resize(-1,256,256,3)

    image= tf.expand_dims(image, axis =0)
# image.shape
# print(image)
# ans=cnn_model.predict(image)
    ans=reconstructed_model.predict(image)
    # ans=ans.tolist()
    print(ans)
    ans=ans.tolist()
    listv = ans[0]
    n = listv.index(max(listv))
    print(n)
    df=pd.read_csv("info.csv")
    name=df.loc[df.sr == n,'Name'].tolist()[0]
    hello=name
    from datetime import datetime

    now = datetime.now() # current date and time
    time = now.strftime("%H:%M:%S")
    print("time:", time)

    year = now.strftime("%Y")
    print("year:", year)

    month = now.strftime("%m")
    print("month:", month)

    day = now.strftime("%d")
    print("day:", day)

    rec=pd.read_csv("Record.csv")
    list=[]
    # list=[name,time,day+month+year]
    list.append(name)
    list.append(time)
    list.append(day+month+year)
    row=pd.Series(list,index=['Name','Clock In Time','Date'])
    print(row)

    rec=rec.append(row, ignore_index=True)
    rec.to_csv("Record.csv",index=False)

# def reconstruct():
reconstructed_model = keras.models.load_model("my_model")

FRAME_WINDOW = st.image([])
if st.button("Clock In",key="69"):
    print("Hi")
    cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
    ret,frame = cap.read()
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
    crop(frame)
    # while(True):
    #     cv2.imshow('img1',frame) #display the captured image
    #     if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y' 
    #         cv2.imwrite('images/c1.png',frame)
    #         cv2.destroyAllWindows()
    #         break

    cap.release()
    st.image(frame)

# st.button(label, key=None, help=None, on_click=None, args=None, kwargs=None, *, disabled=False)

if hello:
    st.write("Hi "+hello)