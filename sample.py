import cv2
import os

count=0
path="Aditi"
def crop(imgName):
    # Read the input image
    print(imgName)
    img = cv2.imread(path+"/"+imgName)

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
        global count
        cv2.imwrite(path+'/'+str(count)+'.jpg', faces)
        count=count+1
	
# Display the output
# cv2.imwrite('detected.jpg', img)
# cv2.imshow("img",img)
    cv2.waitKey()

# print(os.listdir(path))
# print(path)
for i in os.listdir(path):
    # print(filename)
    crop(i)