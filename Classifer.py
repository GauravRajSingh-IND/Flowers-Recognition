import matplotlib.pyplot as plt
from zipfile import ZipFile
from pathlib import Path
from os.path import join
from tqdm import tqdm
import random as rn
import numpy as np
import cv2
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.preprocessing.image import  ImageDataGenerator

filenames = "/content/drive/My Drive/Dog_Cat/8782_44566_bundle_archive.zip"
with ZipFile(filenames, 'r') as zip:
  zip.extractall()
  
  x = []
z = []

IMG_SIZE = 150

daisy_dir = "/content/flowers/daisy"
dandelion_dir = "/content/flowers/dandelion"
flowers_dir = "/content/flowers/flowers"
rose_dir = "/content/flowers/rose"
sunflower_dir = "/content/flowers/sunflower"
tulip_dir = "/content/flowers/tulip"

def assign_label(img, flower_type):
  return flower_type
  
def make_train_data(flower_type, DIR):
  for img in tqdm(os.listdir(DIR)):
    label = assign_label(img, flower_type)
    path = os.path.join(DIR, img)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    x.append(np.array(img))
    z.append(str(label))
   
   
make_train_data('daisy', daisy_dir)
print(len(x))
make_train_data('rose', rose_dir)
print(len(x))
make_train_data('sunflower', sunflower_dir)
make_train_data('tulip', tulip_dir)
print(len(x))

print("Number of Images", len(x))
print("Number of labels", len(z))

plt.imshow(x[1000])
plt.title('Flower: ' +  z[1000])

x = np.array(x, dtype= 'float32') / 255.0
x[0]

le = LabelEncoder()
y = le.fit_transform(z)
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

print("Number of training data", x_train.shape)
print("Number of Label data", y_train.shape)

print("Number of tesing data", x_test.shape)
print("Number of Label data", y_test.shape)


model = Sequential()
model.add(Conv2D(32, kernel_size= (3,3), padding = 'Same', strides= (1, 1), activation= 'relu', input_shape = (150, 150, 3)))
model.add(MaxPooling2D(pool_size= (2,2)))
model.add(Conv2D(32, kernel_size= (3,3), padding = 'Same', strides= (1, 1), activation= 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size= (3,3), padding = 'Same', strides= (1, 1), activation= 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size= (3,3), padding = 'Same', strides= (1, 1), activation= 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
model.add(Conv2D(128, kernel_size= (3,3), padding = 'Same', strides= (1, 1), activation= 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
model.add(Dropout(0.25))

model.add(Dense(128, activation= 'relu'))
model.add(Dense(256, activation= 'relu'))
model.add(Flatten())
model.add(Dense(64, activation= 'relu'))
model.add(Dense(128, activation= 'relu'))
model.add(Dense(256, activation= 'relu'))
model.add(Dense(4, activation= 'softmax'))

model.summary()
model.compile(optimizer= Nadam(lr = 0.001), loss = 'categorical_crossentropy', metrics= ['accuracy'])

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False  # randomly flip images
        )

datagen.fit(x_train)

history = model.fit(datagen.flow(x_train, y_train, batch_size= 128), epochs= 50, validation_data= (x_test, y_test), verbose= 1)

prediction = model.predict(x_test, batch_size = 32)
print(classification_report(y_test.argmax(axis=1),
	prediction.argmax(axis=1), target_names=le.classes_))
  
model.save("FlowerModel", save_format = "h5")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])

