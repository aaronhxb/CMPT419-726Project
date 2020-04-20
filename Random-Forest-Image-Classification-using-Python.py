from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve, GridSearchCV

# In[3]:


# make a fix file size
fixed_size  = tuple((224,224))

#train path 
#train_path = "stanford-car-dataset-by-classes-folder/car_data/car_data/train/"
train_path = "data/train/"
# no of trees for Random Forests
num_tree = 100

# bins for histograms 
bins = 8

# train_test_split size
test_size = 0.20

# seed for reproducing same result 
seed = 42 


# In[4]:


# features description -1:  Hu Moments

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# In[5]:


# feature-descriptor -2 Haralick Texture 

def fd_haralick(image):
    # conver the image to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Ccompute the haralick texture fetature ve tor 
    haralic = mahotas.features.haralick(gray).mean(axis=0)
    return haralic


# In[6]:


# feature-description -3 Color Histogram

def fd_histogram(image, mask=None):
    # conver the image to HSV colors-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #COPUTE THE COLOR HISTPGRAM
    hist  = cv2.calcHist([image],[0,1,2],None,[bins,bins,bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist,hist)
    # return the histog....
    return hist.flatten()


# In[7]:


# get the training data labels 
train_labels = os.listdir(train_path)

# sort the training labesl 
train_labels.sort()
print(train_labels)

# empty list to hold feature vectors and labels 
global_features = []
labels = []

i, j = 0, 0 
k = 0

# num of images per class 
images_per_class = 80


# <h1>loop insise the folder for train images </h1>

# In[84]:


# ittirate the folder to get the image label name

#get_ipython().run_line_magic('time', '')
# lop over the training data sub folder 

for training_name in train_labels:
    # join the training data path and each species training folder
    if training_name =='.DS_Store':
        continue
        #dir = os.path.join(train_path, training_name;)
    dir = train_path + training_name

    # get the current training label
    current_label = training_name

    k = 1
    # loop over the images in each sub-folder
        
    for file in os.listdir(dir):


        file = dir + "/"  + file
       
        # read the image and resize it to a fixed-size
        image = cv2.imread(file) 
        
        if image is not None:
            image = cv2.resize(image,fixed_size)
            fv_hu_moments = fd_hu_moments(image)
            fv_haralick   = fd_haralick(image)
            fv_histogram  = fd_histogram(image)

        # Concatenate global features
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

        i += 1
        k += 1
    print("[STATUS] processed folder: {}".format(current_label))
    j += 1

print("[STATUS] completed Global Feature Extraction...")




# In[10]:


# import the feature vector and trained labels

#h5f_data = h5py.File('output/data.h5', 'r')
#h5f_label = h5py.File('output/labels.h5', 'r')
#
#global_features_string = h5f_data['dataset_1']
#global_labels_string = h5f_label['dataset_1']

#global_features = np.array(global_features_string)
global_features = np.array(global_features)
#global_labels = np.array(global_labels_string)

global_labels = np.array(labels)



# In[11]:

print("global_features ", global_features)

# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)


# <h3>RandomForest</h3>

# In[128]:
from sklearn.metrics import classification_report

# create the model - Random Forests
clf  = RandomForestClassifier(n_estimators= 1800, max_features='sqrt', max_depth=80, 
                                min_samples_split=5, min_samples_leaf=2,bootstrap=True, oob_score = True)


# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)

#print(clf.fit(trainDataGlobal, trainLabelsGlobal))

clf_pred = clf.predict(trainDataGlobal)
#clf_pred = clf.predict(global_feature.reshape(1,-1))[0]
print(classification_report(trainLabelsGlobal,clf_pred))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(trainLabelsGlobal,clf_pred))
print("Accuracy:", accuracy_score(trainLabelsGlobal,clf_pred))
#print(clf.predict(trainDataGlobal))
#print(trainLabelsGlobal)
#print(clf.predict(global_feature.reshape(1,-1))[0])
print (clf.oob_score_)
#clf.fit(testDataGlobal, testLabelsGlobal)
y_pred = clf.predict(testDataGlobal)
print(classification_report(testLabelsGlobal,y_pred))
print(confusion_matrix(testLabelsGlobal,y_pred))
#
print("Accuracy:", accuracy_score(testLabelsGlobal,y_pred))
# In[129]:
from sklearn.metrics import auc
#print("auc:", auc(testLabelsGlobal,y_pred))

#rf2 = RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=120,
#                                  min_samples_leaf=20,max_features=7 ,oob_score=True, random_state=10)
#rf2.fit(trainDataGlobal,trainLabelsGlobal)
#print (rf2.oob_score_)
#clf_pred = rf2.predict(trainDataGlobal)
#
#print("Accuracy:", accuracy_score(trainLabelsGlobal,clf_pred))


clf2 = GridSearchCV(clf, param_grid={'n_estimators':[600,1400],'min_samples_leaf':[2,5], 
                                        'min_samples_split':[2,10], 
                                        'max_features' : ['sqrt'],
                                        'bootstrap': [True, False], 
                                         'max_depth' : [40,80]})
model = clf2.fit(trainDataGlobal,trainLabelsGlobal)
#y_pred_train = model.predict(trainDataGlobal)
#    # predictions for test
#y_pred_test = model.predict(testDataGlobal)
#    # training metrics
#print("Training metrics:")
#print(classification_report(y_true= trainLabelsGlobal, y_pred= y_pred_train))
#    
#    # test data metrics
#print("Test data metrics:")
#print(classification_report(y_true= testLabelsGlobal, y_pred= y_pred_test))
# Predictions on testset
#y_pred_test = model.predict(X_final_test)
    # test data metrics
#print("Test data metrics:")
#print(classification_report(y_true= y_final_test, y_pred= y_pred_test))

#clf2 = GridSearchCV(clf, param_grid={'n_estimators':[100,200],'min_samples_leaf':[2,3]})
#model = clf2.fit(trainDataGlobal, trainLabelsGlobal)
#
#
#y_pred_train = model.predict(trainDataGlobal)
#    # predictions for test
#y_pred_test = model.predict(testDataGlobal)
#    # training metrics
#print("Training metrics:")
#print(classification_report(y_true= trainLabelsGlobal, y_pred= y_pred_train))
#    
#    # test data metrics
#print("Test data metrics:")
#print(classification_report(y_true= testLabelsGlobal, y_pred= y_pred_test))



# path to test data
test_path = "stanford-car-dataset-by-classes-folder/car_data/car_data/test/Acura Integra Type R 2001"

# loop through the test images
#for file in glob.glob(test_path + "/*.jpg"):
#for file in os.listdir(test_path):    
#
#    file = test_path + "/" + file
#    #print(file)
#    
#    # read the image
#    image = cv2.imread(file)
#
#    # resize the image
#    image = cv2.resize(image, fixed_size)
#
#    # Global Feature extraction
#    fv_hu_moments = fd_hu_moments(image)
#    fv_haralick   = fd_haralick(image)
#    fv_histogram  = fd_histogram(image)
#
#    # Concatenate global features
#
#    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
#
#    # predict label of test image
#    prediction = clf.predict(global_feature.reshape(1,-1))[0]
#
#    # show predicted label on image
#    cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
#
#    # display the output image
#    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#    plt.show()
#


# In[ ]:





# In[109]:


#rfc_pred = rfc.predict(trainDataGlobal)


# In[108]:


#print(confusion_matrix(trainLabelsGlobal,rfc_pred))


# In[107]:


#print(classification_report(trainLabelsGlobal,rfc_pred))