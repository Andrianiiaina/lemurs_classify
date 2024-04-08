import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import cv2

#lien de l'image à tester
link='test\image.jpg'

class_names=["eulemur_fulvus","fossa","hapalemur_griseus","lemur_catta","propitheque_coquerel","propitheque_couronne","indri-indri"]
model=tf.keras.models.load_model('image_oo2.model')


img=cv2.imread(link)
img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#plt.imshow(img, cmap=plt.cm.binary)
prediction = model.predict(np.array([img])/255)

index= np.argmax(prediction)
print(prediction)
print(class_names[index])
print(prediction[0][index]*100)
#plt.show()