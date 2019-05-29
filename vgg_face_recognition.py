
# coding: utf-8

# In[1]:


from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt


# # CNN Architecture

# In[2]:


model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))
#model.summary()


# # Adding weights to the model

# In[3]:


model.load_weights('C:/Users/Pavan Raju/Desktop/FACE-RECOGNITION/FaceRecognitoin/vgg_face_weights.h5')


# In[4]:


vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)


# # Creating necessary functions

# In[5]:


#to load images from the path
def load_image(path):
    img = load_img(path, target_size=(224,224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img)
    return img
#to load images from webcam
def webcam_image(image):
    img = cv2.resize(image,(224,224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def EuclideanDistance(input1,input2):
    embedding = input1- input2
    euc_sum = np.sum(np.multiply(embedding,embedding))
    euc_dist = np.sqrt(euc_sum)
    return euc_dist

def CousineDistance(input1,input2):
    a  = np.matmul(np.transpose(input1),input2)
    b  = np.sum(np.multiply(input1,input1))
    c  = np.sum(np.multiply(input2,input2))
    distance = 1 - (a/(np.sqrt(b)*np.sqrt(c)))
    return distance




# In[37]:


#Create embedded files of the database
import os

def create_database(path):
    for file_name in os.listdir(path):
        name,ext = os.path.splitext(file_name)
        employees[name] = vgg_face_descriptor.predict(load_image(os.path.join(path,file_name)))[0,:]
    return employees

database_path = "C:/Users/Pavan Raju/Desktop/FACE-RECOGNITION/FaceRecognitoin/dataset/"
employees = dict()   
database = create_database(database_path)
database


# In[38]:


import cv2
import requests


#url = "http://mmqn6w:9779@ipphonecamera.deskshare.com/"
#url = "http://192.168.43.1:8080/"

'''json_data={"username":"mmqn6w","password":"9779","language":"en"}
headers={
    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8*',
    'Accept-Encoding':'gzip, deflate',
    'Accept-Language': 'en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7',
    'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36',
    'Referer': 'http://ipphonecamera.deskshare.com/'
}'''

#url = "http://netra:12345@10.194.178.222:8080/shot.jpg"

def webcam_on(database):
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("http://netra:12345@10.194.178.222:8080/shot.jpg")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while True:
        ret,img = cap.read()
        #img_req = requests.get(url)
        #img_arr = np.array(bytearray(img_req.content), dtype=np.uint8)
        #img = cv2.imdecode(img_arr, -1)
        faces = face_cascade.detectMultiScale(img,1.3,5)
        for (x,y,w,h) in faces:
            x1 = x
            y1 = y
            x2 = x+w  
            y2 = y+h 
            img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            detected_face = img[int(y1):int(y2), int(x1):int(x2)]
            detected_face = webcam_image(detected_face)
            captured_embedding = vgg_face_descriptor.predict(detected_face)[0,:] 
            #encode = base64.b64encode(img)
            #client.publish("netra123/test123",encode)
            for i in database:
                person_name = i
                person_embedding = database[person_name]
                euc_distance = EuclideanDistance(person_embedding,captured_embedding)
                cos_distance = CousineDistance(person_embedding,captured_embedding)
                euc_square = euc_distance*euc_distance
                cos_euc = cos_distance*euc_distance
                print("euc_distance : ",euc_distance)
                print("cos_distance : ",cos_distance)
                print("C&E distance : ",cos_euc)
                if cos_distance < 0.23 and euc_distance < 75 and cos_euc < 40:
                    print("Person name is : ",person_name)
                    print("      Euclidean_distance is",euc_distance)
                    print("      Cosine_distance is",cos_distance)
                    print("      cos_euc",cos_euc)
                    cv2.putText(img, person_name ,(x1+5,y1-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                else:
                    img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
                #    cv2.putText(img,"Unable to identify",(x1+5,y1-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            #key = cv2.waitKey(100)
        cv2.imshow("recognizer",img)

        if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
            break
    cap.release()
    cv2.destroyAllWindows()


# In[39]:


webcam_on(database)


"""
import paho.mqtt.client as mqtt
import base64
from PIL import Image
import io
import cv2
import numpy as np


def on_connect(client,userdata,flags,rc):
    print("connected with code : ",str(rc))
    client.subscribe("netra123/query")

def on_message(client,userdata,msg):
    decode = base64.b64decode(msg.payload)
    print("***")
    img = Image.open(io.BytesIO(decode))
    img = np.array(img)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    captured_img = webcam_image(img)
    captured_emb = vgg_face_descriptor.predict(captured_img)[0,:]
    for i in database:
        person_name = i
        person_embedding = database[person_name][0:]
        euc_distance = EuclideanDistance(person_embedding,captured_emb)
        cos_distance = CousineDistance(person_embedding,captured_emb)
        if cos_distance < 0.3 :
            #print("Euclidean_distance = ",euc_distance)
            #print("Cosine_distance = ",cos_distance)
            print("Detected face is ", person_name)
            client.publish("netra123/result",person_name)
    
    
client = mqtt.Client()
client.connect("broker.hivemq.com",1883,60)
client.on_connect = on_connect
client.on_message = on_message
client.loop_forever()

client = mqtt.Client()
client.connect("broker.hivemq.com",1883,60)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while cap.isOpened():
    ret, img = cap.read()
    faces = face_cascade.detectMultiScale(img,1.3,5)
    for (x,y,w,h) in faces:
        x1 = x
        y1 = y
        x2 = x+w  
        y2 = y+h 
        img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
        detected_face = img[int(y1):int(y2), int(x1):int(x2)]
        cv2.imwrite("sample.jpg",detected_face)
        encode_image = open("sample.jpg",'rb')
        output_image = encode_image.read()
        output = base64.b64encode(output_image)
        client.publish("netra123/test123",output)
    cv2.imshow("image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
"""