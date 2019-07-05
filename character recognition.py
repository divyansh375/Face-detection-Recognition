import pytesseract
from PIL import Image
import requests
import time

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'
a = Image.open('car.jpeg')


print(pytesseract.image_to_boxes(a))

loc_x=10
loc_y=10

import cv2
import numpy as np



url = ""	

ti=0
t=0
tim=0

ti=0

cap = cv2.VideoCapture('v.avi') 

while (t<10) :  
  
     
    #ret, img = cap.read()
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content) , dtype = np.uint8)
    img = cv2.imdecode(img_arr , -1)
    
  


            

        
        
        
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    start = time.clock()
    
    n=pytesseract.image_to_string(img)
    ti= ti+(time.clock() - start)
    tim=tim+ti
    ti=0
    print("frame")
    print(t)
    t=t+1
    


    n=n.replace("wwwBANDICAM",'')
    n=n.replace("BANDICAME",'')
    n=n.replace("Www",'')
    n=n.replace("??",'')
    n=n.replace("Kelson",'')
    n=n.replace("mE",'')
    n=n.replace("com",'')
    n=n.replace(")",'')

    n = [c for c in n if c.isalpha()] 
    n = ' '.join(n)

    
    cv2.putText(img, str(n) , (loc_x,loc_y),font, 0.8, (255 ,0 , 0), 2, cv2.LINE_AA)
    loc_x=loc_x+10
    loc_y=loc_y+10
    if(loc_x==100):
        loc_x=10
        loc_y=10
    print(n)
    print("\n")



        
        
           
        
    
  
        
        
         
  
    
    cv2.imshow('img',img) 
  
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
  
print("avarage time taken for each frame = ")
print(tim/10)
cap.release() 
  

cv2.destroyAllWindows() 

