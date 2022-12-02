import cv2 #import cv2 library
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


#load the data set file
with_mask = np.load(r"C:\Users\Py.Developer_Aniket\Desktop\AI Python\Mini_Project\Face_Mask_Detection_Machine_Learning\with_mask.npy")
without_mask =np.load(r"C:\Users\Py.Developer_Aniket\Desktop\AI Python\Mini_Project\Face_Mask_Detection_Machine_Learning\without_mask.npy")

#check the size of data_set
#
print(with_mask.shape)
print(without_mask.shape)
#

#convert data set into 2 diamentional array
with_mask = with_mask.reshape(100,50*50*3)
without_mask = without_mask.reshape(100,50*50*3)

#check the size of data_set
#
print(with_mask.shape)
print(without_mask.shape)
#

#combined two dataset
x = np.r_[with_mask,without_mask]
#print(x .shape)

#give label to your data
#now we have to label the dataset
with_mask  = 0
without_mask =1
label =np.zeros(x.shape[0])
label[100:]=1.0

x_train,x_test,y_train,y_test =train_test_split(x,label,test_size = 0.40)

svm = SVC()
svm.fit(x_train,y_train)
y_pred = svm.predict(x_test)
#print the accuracy of your data
#accracy below 1 should be good
print(accuracy_score(y_test,y_pred))

