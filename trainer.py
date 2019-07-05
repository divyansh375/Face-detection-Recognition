import cv2
import numpy as np
import requests

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

url = ""
c=0
while (c<50):


    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content) , dtype = np.uint8)
    img = cv2.imdecode(img_arr , -1)
  
     
    
  
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 

    for (x,y,w,h) in faces: 
        
         
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w]
        resize_img = cv2.resize(roi_gray  , (192 , 192))
        s=""
        s= s + str(c)
        s=s+".jpg"
        cv2.imwrite(s,resize_img)
    c=c+1



    

        
