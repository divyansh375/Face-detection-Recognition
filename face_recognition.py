import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense 
from keras.utils import to_categorical
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
import keras




x_train=[]
y_train=[]
for i in range(1,11,1):
    x="trainer1/"+str(i)+".jpg"
    v=cv2.imread(x)
    
    v = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
    v = cv2.resize(v  , (192 , 192))
    x_train.append(v)
    y_train.append(1)



for i in range(1,11,1):
    x="trainer2/"+str(i)+".jpg"
    v=cv2.imread(x)
    v = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY) 
    v = cv2.resize(v  , (192 , 192))

    x_train.append(v)
    y_train.append(2)




x_train = np.array(x_train)

y_train = np.array(y_train)

y_train = to_categorical(y_train)

x_train = x_train.reshape(-1,192,192,1)


x_train = x_train.astype('float32')

x_train =  x_train / 255.




# building the convolution neural network 

model = Sequential()

#1st convolution layer has 32 nodes 
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(192,192,1),padding='same'))
model.add(MaxPooling2D((2, 2),padding='same'))
#2nd convolution layer has 64 nodes
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
#Reducing the dimension to 1
model.add(Flatten())
model.add(Dense(64, activation='linear'))                  
model.add(Dense(3, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 10)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])




 
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml') 


  

cap = cv2.VideoCapture('virat_and_anushka.avi') 
  

while 1:  
  
     
    ret, img = cap.read()  
  
    
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
        
        
        if(c==1 and p ==1):
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
            print("virat")
            cv2.putText(img, 'virat', (x,y), font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            print(p)
            print("\n")

        elif(c==2 and p == 1 ):
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
            print("anushka")
            cv2.putText(img, 'anushka', (x,y), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            print(p)
            print("\n")
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            
            cv2.putText(img, 'unknown', (x,y), font, 0.8, (0,0, 255), 2, cv2.LINE_AA)
            print("not detected")
            print(p)
            print("\n")
            
           
        
    
  
        
        
         
  
    
    cv2.imshow('img',img) 
  
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
  

cap.release() 
  

cv2.destroyAllWindows() 

