import numpy as np
import cv2
import time

#xml for detecting face
faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

# Create our shapening kernel, it must equal to one eventually
kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])
num=0
while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    img = cv2.flip(img,1)

    # Our operations on the frame come here
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2,minNeighbors=5,minSize=(30,30),)
    
 
    if faces != '()': # condition for webcam does not detect person's face 
        for (x,y,w,h) in faces:
            sharpened = cv2.filter2D(img, -1, kernel_sharpening) #sharpen image which has captured
            cv2.imwrite(str(num)+'.jpg',img[y:y+h, x:x+w])
            #time.sleep(0.3) #sleep for 0.3seconds for running code
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),0) # (255,0,0) is color for the rectangle, and the number 0 next to the rectangle color is thickness of line
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            num  = num+1    

    # Display the resulting frame
    cv2.imshow('frame',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
