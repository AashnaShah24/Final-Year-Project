import cv2
import os
import pandas as pd
import streamlit as st

cam = cv2.VideoCapture(0)
st.title('New User Registration')

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
imagecounter=0
# For each person, enter one numeric face id
# face_id = st.text_input('\n Make sure the first user entered is 0.enter user id end press <return> ==>  ')
face_id=0
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
mainFolder="a"
count=0
if not os.path.exists(mainFolder+"/0"):
# if os.listdir(mainFolder)== Null :
    count = 0
else:
    count=len(os.listdir(mainFolder))
# Creating a folder
os.mkdir("a/"+str(count))
form = st.form(key='Enter details')
name = form.text_input('Enter your name')
number=form.text_input('Enter your mobile Number')
submit = form.form_submit_button('Submit')

def informationSaver():
    info=pd.read_csv("Info.csv",index_col="sr").astype(str)
    info=info.dropna()
    lst={"Name":name,"Mobile Number":number}
    info=info.append(lst,ignore_index=True)
    info.reset_index().rename(columns={"index":"sr"}).to_csv("Info.csv", index=False)

if submit:

    informationSaver()

    
    while(True):

        ret, img = cam.read()
        img = cv2.flip(img, 1) # flip video image vertically
        # colour = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        colour=img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w+50,y+h+50), (255,0,0), 2)   
            # global imagecounter  
            imagecounter += 1

            # Save the captured image into the datasets folder
            gray = gray[y:y+h,x:x+w]
            colour=colour[y:y+h,x:x+w]
            
            
            # cv2.imwrite("d/" + str(count) + '/' + str(imagecounter) + ".jpg",gray )
            cv2.imwrite("a/" + str(count) + '/' + str(imagecounter) + ".jpg",colour)

            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif imagecounter >= 70: # Take 70 face sample and stop video
            break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()