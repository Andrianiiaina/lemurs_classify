import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

targets=[]
features=[]

#les images de train sont dans le docs lemurs_park_images
#les images de test sont dans le docs lemrs_park_test
files=os.listdir('lemurs_park_images')
X=[]
for file in files:
    images=glob.glob("lemurs_park_images\\"+file+"\\*.jpg")+glob.glob("lemurs_park_images\\"+file+"\\*.jpeg")
    for image in images:
        X.append(image)
random.shuffle(X)

datagen= ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True, fill_mode='nearest')
def get_target(file):
    if "fulvus" in file:
        return [1,0,0,0,0,0,0]
    elif "fossa" in file:
        return [0,1,0,0,0,0,0]
    elif "griseus" in file:
        return [0,0,1,0,0,0,0]   
    elif "maki" in file:
        return [0,0,0,1,0,0,0]
    elif "coquerel" in file:
        return [0,0,0,0,1,0,0]
    elif "coronatus" in file:
        return [0,0,0,0,0,1,0]
    elif "indri" in file:
        return [0,0,0,0,0,0,1]

for image in X:
    for j in range(10):
        img=np.array(Image.open(image).resize((224,224)))/255.0
        aug_image=datagen.random_transform(img)
        features.append(aug_image)
        targets.append(get_target(image))
            


features=np.array(features)
targets=np.array(targets) 

#augmentation des datas
data_augmentation= ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True, fill_mode='nearest')
test_generator=data_augmentation.flow_from_directory('lemurs_park_test',target_size=(224,224),batch_size=15,class_mode='categorical')
X_test,Y_test= test_generator.next()


# Load le pretrained MobileNetV2 model without top classification layers
base_model = MobileNetV2(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(7, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)


# Freeze the layers of the pretrained model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train le model (fine-tuning the top layers)
#model.fit(X_train,Y_train, epochs=10)
history=model.fit(features,targets, epochs=10,validation_data=(X_test,Y_test))

model.save('image_oo2.model')
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoque')
plt.ylabel('Accuracy')
plt.ylabel([0.1,1])
plt.legend(loc='lower right')
plt.show()

