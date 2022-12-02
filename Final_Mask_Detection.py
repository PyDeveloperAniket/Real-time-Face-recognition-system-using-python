import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#load xml feature file so that we can detect face from image 
'''It is an Object Detection Algorithm used to identify faces in an image or a real time video. 
The algorithm uses edge or line detection features proposed by Viola and Jones in their research paper 
“Rapid Object Detection using a Boosted Cascade of Simple Features” published in 2001 '''
haar_data = cv2.CascadeClassifier("C:/Users/Py.Developer_Aniket/Desktop/AI Python/Mini_Project/Face_Mask_Detection_Machine_Learning/haarcascade_frontalface_default.xml")
#load the data set file
with_mask = np.load("C:/Users/Py.Developer_Aniket/Desktop/AI Python/Mini_Project/Face_Mask_Detection_Machine_Learning/with_mask.npy")
without_mask = np.load("C:/Users/Py.Developer_Aniket/Desktop/AI Python/Mini_Project/Face_Mask_Detection_Machine_Learning/without_mask.npy")

#convert data set into 2 diamentional array
with_mask = with_mask.reshape(100,50*50*3)
without_mask = without_mask.reshape(100,50*50*3)

#combined two dataset
x = np.r_[with_mask,without_mask]
#print(x.shape)

#we are giving a label to our data
#now we have to label the dataset
#with_mask  = 0
#without_mask =1
label = np.zeros(x.shape[0])
label[100:]=1.0

x_train,x_test,y_train,y_test =train_test_split(x,label,test_size = 0.40)

svm = SVC()
svm.fit(x_train,y_train)
y_pred = svm.predict(x_test)
#print the accuracy of your data
#accracy below 1 should be good
# print(accuracy_score(y_test,y_pred))

#create variant to display mask and no mask string 
names = {0 : "Mask Detected", 1 :"Mask Not Detected"}

#font for name
font = cv2.FONT_HERSHEY_COMPLEX = 2
#to start the camera
capture = cv2.VideoCapture(0)

while(1):
    #load the image from camera
    flag,img = capture.read()
    #check camera is working or not
    if flag :
        #this function return array for faces from image 
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            #draw rectangle over face
            #cv2.rectangle(img,(x,y),(w,h),(r,g,b),boarder_thickness)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,200),2)
            #slice the face from image
            face = img[y:y+w,x:x+w,:]
            #resize the face into one size
            face = cv2.resize(face,(50,50))
            face = face.reshape(1,-1)
            pred = svm.predict(face)
            n = names[int(pred)]
            print(n)
            cv2.putText(img,n,(x,y),font,1,(255,100,100),2) #show image
        cv2.imshow("output",img) #show image
        # 27 is ASCII value of escapse key
        #It is use to run thread which waiting for ESC button becuase of this image will show on window
        #if you not use this function then it will not load image
        if((cv2.waitKey(2) == 27)):
            break;
cv2.destroyAllWindows()
