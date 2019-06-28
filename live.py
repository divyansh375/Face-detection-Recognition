import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten , MaxPooling2D , Dropout
from keras.utils import to_categorical
from keras.layers.advanced_activations import LeakyReLU
import keras
import requests



x_train=[]
y_train=[]

for i in range(24,49,1):
    x="trainer1/"+str(i)+".jpg"
    v=cv2.imread(x)
      
    
    v = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
    v = cv2.resize(v  , (192 , 192))
    x_train.append(v)
    y_train.append(1)

print("ok")


for i in range(24,49,1):
    x="trainer2/"+str(i)+".jpg"
    v=cv2.imread(x)
    v = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY) 
    v = cv2.resize(v  , (192 , 192))

    x_train.append(v)
    y_train.append(2)

print("ok")

for i in range(18,35,1):
    x="trainer3/"+str(i)+".jpg"
    v=cv2.imread(x)
    v = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY) 
    v = cv2.resize(v  , (192 , 192))

    x_train.append(v)
    y_train.append(3)

print("ok")


    


x_train = np.array(x_train)

y_train = np.array(y_train)

y_train = to_categorical(y_train)

x_train = x_train.reshape(-1,192,192,1)


x_train = x_train.astype('float32')

x_train =  x_train / 255.


model = Sequential()

#1st layer 
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(192,192,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))

#2nd layer : reduce the dimensionality of the feature map
model.add(MaxPooling2D((2, 2),padding='same'))

#3rd layer
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))

#4th layer
model.add(MaxPooling2D(pool_size= (2, 2),padding='same'))

#5th layer
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))

#6th layer
model.add(MaxPooling2D(pool_size= (2, 2),padding='same'))

#7th layer : reduce the dimensionality to one
model.add(Flatten())

#8th layer : pass through dense layer
model.add(Dense(128, activation= 'linear'))
model.add(LeakyReLU(alpha=0.1))                  

#9th layer : final layer
model.add(Dense(4, activation= 'softmax'))




model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )
print(model.summary())

model.fit(x_train, y_train, epochs = 5)



url =  ""

 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

#the loop below takes  
  


  

while 1:  
  
     
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content) , dtype = np.uint8)
    img = cv2.imdecode(img_arr , -1)
  
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
  
    for (x,y,w,h) in faces: 
        
         
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w]
        resize_img = cv2.resize(roi_gray  , (192 , 192))
        cv2.imshow('s',resize_img)
        x_test=[]
        x_test.append(resize_img)
        x_test = np.array(x_test)
        x_test = x_test.reshape(1,192,192,1)
        font = cv2.FONT_HERSHEY_SIMPLEX

        c=model.predict(x_test)
        p= max(model.predict(x_test))
        p= max(p)
        c = np.argmax(np.round(c),axis=1)
        
        
        if(c==1):
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
            print("")
            cv2.putText(img, '', (x,y), font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            print(p)
            print("\n")



        elif(c==2  ):
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            print("")
            cv2.putText(img, '', (x,y), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            print(p)
            print("\n")
            
        
        elif(c==3 ):
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
            print("")
            cv2.putText(img, '', (x,y), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            print(p)
            print("\n")            
           
        
    
  
        
        
         
  
    img =  cv2.resize(img  , (900 , 700))
    cv2.imshow('img',img) 
  
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
  

cap.release() 
  

cv2.destroyAllWindows() 

