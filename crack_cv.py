# %%
#1. Import Packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,callbacks,applications,models
import numpy as np
import matplotlib.pyplot as plt
import os,datetime
import keras.api._v2.keras as keras
import cv2
import imghdr

# %%
#name of the folder contain training datasets
data_dir='Concrete Crack Images for Classification'

image_exts=['jpeg','jpg','bmp','png']

# %%
#check/inspect images in the path folder
for image_class in os.listdir(data_dir):
    print(image_class)
    for image in os.listdir(os.path.join(data_dir,image_class)):
        image_path=os.path.join(data_dir,image_class,image)
        try:
            img=cv2.imread(image_path)
            tip=imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))

# %%
#load data
data=keras.utils.image_dataset_from_directory(data_dir, label_mode='binary', seed=1337)
#
data_iterator=data.as_numpy_iterator()

batch=data_iterator.next()

#%%
#Examine the pet food aka batch
fig, ax=plt.subplots(ncols=4,figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

#%%
#Scale Data
print(batch[0][1])

data=data.map(lambda x,y: (x/255, y))

data.as_numpy_iterator().next()

#%%
#Split Data 
train_size=int(len(data)*0.7)
val_size=int(len(data)*0.2)  #during training to tune parametes like hidden layer
test_size=int(len(data)*0.1) #after training, to see the performance of model

#%%
#Train/Val/Test
train=data.take(train_size) 
val=data.skip(train_size).take(val_size)
test=data.skip(train_size+val_size).take(test_size)

#%%
#Build model learning
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model=Sequential()

model.add(Conv2D(16,(3,3),1,activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(16,(3,3),1,activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16,(3,3),1,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss=tf.losses.binary_crossentropy,metrics=['accuracy'])

#%%
#To see overall structure of our model
model.summary() 

#%%
# Create the TensorBoard callback object
PATH = os.getcwd()
logpath = os.path.join(PATH,'tensorboard_log',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = callbacks.TensorBoard(logpath)

#%%
#Model training
early_stopping = callbacks.EarlyStopping(patience=2)
EPOCHS = 5
hist = model.fit(train,validation_data=val,epochs=EPOCHS,callbacks=[tb,early_stopping])

#%%
#Model evaluate after training
model.evaluate(test)

#%%
#Plot performance Loss
fig=plt.figure()
plt.plot(hist.history['loss'],color='teal',label='loss')
plt.plot(hist.history['val_loss'],color='orange',label='val_loss')
fig.suptitle('Loss',fontsize=20)
plt.legend(loc='upper left')
plt.show()

#%%
##Plot performance Accuracy
fig=plt.figure()
plt.plot(hist.history['accuracy'],color='teal',label='accuracy')
plt.plot(hist.history['val_accuracy'],color='orange',label='val_accuracy')
fig.suptitle('Accuracy',fontsize=20)
plt.legend(loc='upper left')
plt.show()

#%%
#Model evaluate performance
results = model.evaluate(test)
print("Test loss:", results[0])
print("Test accuracy:", results[1])

#%%
#Evaluate Result
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

pre=Precision()
re=Recall()
acc=BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X,y=batch
    pred=model.predict(X)
    pre.update_state(y,pred)
    re.update_state(y,pred)
    acc.update_state(y,pred)

print(pre.result(),re.result(),acc.result())

#%%
#Predict Probability
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix,classification_report

y_pred=model.predict(val)

y_pred=np.where(y_pred>0.5,1,0) #is in probability, needs to convert to category
y_test = tf.concat([y for x,y in val], axis=0) #y_test=val

#%%
#Classification Report
print(classification_report(y_test,y_pred))

#%%
#Confusion Matrix
cm=confusion_matrix(y_test,y_pred,normalize='pred')
print(cm)

#%%
#Plot Confusion MAtrix
display=ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot(cmap=plt.cm.Blues)
plt.show()

#%%
#Test Model Learning
import cv2

img=cv2.imread('crack001.jpg') #place the image u wan to test
plt.imshow(img)
plt.show()

#%%
#Resize the image equivalent to dataset
resize=tf.image.resize(img,(256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()

#%%
#Prediction making
y_pred=model.predict(np.expand_dims(resize/256,0))

if y_pred > 0.5:
    print('Predicted class is Crack')
else:
    print('Predict class is Not Crack')

#%%
#Save the model
from tensorflow.keras.models import load_model

model.save(os.path.join('models','crack_classify_v1.h5'))

# %%
img=cv2.imread('goodwall001.jpg') #place the image u wan to test
plt.imshow(img)
plt.show()

#%%
#Resize the image equivalent to dataset
resize=tf.image.resize(img,(256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()

#%%
#Prediction making
y_pred=model.predict(np.expand_dims(resize/256,0))

if y_pred > 0.5:
    print('Predicted class is Crack')
else:
    print('Predict class is Not Crack')

# %%
