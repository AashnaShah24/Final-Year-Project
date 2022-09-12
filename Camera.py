import cv2
import streamlit as st

st.title("Webcam Live Feed")
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

def crop(img):
    # Read the input image
    # print(imgName)
    # img = cv2.imread("Images/"+imgName)

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faces = img[y:y + h, x:x + w]
        faces=cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)
        # cv2.imshow('face',faces)
        st.image(faces,channels="BGR")
        # global count
        cv2.imwrite(''+"1"+'.jpg', faces)
        # count=count+1



if st.button("Clock In",key="0"):
    # while True:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # FRAME_WINDOW.image(frame)
    camera.release()
    st.image(frame)
    crop(frame)









# import cv2
# import streamlit as st

# st.title("Webcam Live Feed")
# run = st.checkbox('Run')
# FRAME_WINDOW = st.image([])
# camera = cv2.VideoCapture(0)

# while run:
#     _, frame = camera.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     FRAME_WINDOW.image(frame)
# else:
#     st.write('Stopped')