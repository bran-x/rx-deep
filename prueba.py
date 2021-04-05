import numpy as np
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model

w, h = 100, 100
model='./modelo/xray_model.h5'
weights='./modelo/xray_weights.h5'

cnn=load_model(model)

cnn.load_weights(weights)

def predecir(img):
    x=load_img(img, target_size=(w, h))
    x=img_to_array(x)
    x=np.expand_dims(x, axis=0)
    prediccion = cnn.predict(x) #Retorna arreglo de 2 dimensiones [[0,0]]
    resultado=prediccion[0]
    print(resultado)
    resultado=np.argmax(resultado)
    if resultado==0:
        print('NORMAL')
    else:
        print('PNEUMONIA')


print("Prediccion 1: (Normal)")
predecir('normal1.jpeg')

print("Prediccion 2: (Pneumonia)")
predecir('pneumonia1.jpeg')