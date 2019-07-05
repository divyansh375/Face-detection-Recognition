from imageai.Detection import ObjectDetection
import cv2 
import requests
import numpy as np
import time

detector =ObjectDetection()

detector.setModelTypeAsYOLOv3()
detector.setModelPath("yolo.h5")
detector.loadModel(detection_speed = "normal" )

url = ""

t=0

tim=0
ti=0

while t<10:  
  
     
    #ret, img = cap.read()
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content) , dtype = np.uint8)
    img = cv2.imdecode(img_arr , -1)
  
    
    
    
    start = time.clock()  
    custom = detector.CustomObjects(person=True)
    ti= ti+(time.clock() - start)
    tim=tim+ti
    ti=0
    
    
    
        
    returned_image, detections = detector.detectObjectsFromImage(input_image=img, output_type="array", minimum_percentage_probability=30 , input_type="array" )     
  
    
    cv2.imshow('img',returned_image)
    s=""
    s=s+"frame"
    s=s+str(t)
    print(s)
    t=t+1
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
  
print("average time taken =")
print(tim/10)
cap.release() 
  

cv2.destroyAllWindows() 
