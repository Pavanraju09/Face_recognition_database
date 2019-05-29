from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt


# # CNN Architecture

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

model.load_weights('C:/Users/Pavan Raju/Desktop/FACE-RECOGNITION/FaceRecognitoin/vgg_face_weights.h5')

vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)


# # Creating necessary functions

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

#Creating a function for webcam detection

def webcam_on(database):
    cap = cv2.VideoCapture(0)
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while True:
        ret,img = cap.read()
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

webcam_on(database)


