data_train = './data/train'
data_test = './data/test'
data_val = './data/val'

epoch = 5
w, h = 100, 100
batch_size = 32
steps = 75
steps_val = 25

filtro_cv_1=64
filtro_cv_2=64
kernel1_tam = (3,3)
kernel2_tam = (3,3)
pool_tam = (2,2)
clases = 2
lr = 0.001

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)
#rescale    escala bits
#shear_range   inclina imagenes
#zoom_range acercamiento de imagenes aleatorio
#horizontal_flip inversion horizontal de imagen
val_datagen = ImageDataGenerator(
    rescale=1./255,
)

img_train = train_datagen.flow_from_directory(
    data_train,
    target_size=(w,h),
    batch_size = batch_size,
    class_mode = 'categorical'
)
img_val = val_datagen.flow_from_directory(
    data_val,
    target_size=(w,h),
    batch_size = batch_size,
    class_mode = 'categorical'
)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
#Creacion de la red neuronal
cnn = Sequential()

#Primera capa
cnn.add(Convolution2D(filtro_cv_1, kernel1_tam, padding='same', input_shape=(w, h, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=pool_tam))

#Segunda capa
cnn.add(Convolution2D(filtro_cv_1, kernel1_tam, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=pool_tam))

cnn.add(Flatten()) #Convierte las caracteristicas a lineales para pasarlas a la red neuronal
cnn.add(Dense(256, activation='relu')) #Numero de neuronas
cnn.add(Dropout(0.5)) #Apagar neuronas para obtener multiples caminos
cnn.add(Dense(clases, activation='softmax')) #Capa de clasificacion

from tensorflow.python.keras import optimizers
#Parametros de optimizacion
cnn.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['accuracy'])


import sys
import os
#Entrenamos el algoritmo
cnn.fit(img_train, steps_per_epoch=steps, epochs= epoch, validation_data=img_val, validation_steps=steps_val)

directorio = './modelo/'
if not os.path.exists(directorio):
    os.mkdir(directorio)
cnn.save('./modelo/xray_model.h5')
cnn.save_weights('./modelo/xray_weights.h5')