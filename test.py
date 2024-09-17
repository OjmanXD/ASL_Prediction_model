import keras
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array


alphabet=['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
model = keras.models.load_model("model2.h5") # can change to model1.h5 for another model
cap = cv2.VideoCapture(0)

def classify(image):
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    proba=model.predict(image)
    idx = np.argmax(proba)
    return alphabet[idx]

cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    image = cv2.imread(r'archive (1)\amer_sign2.png')
    cv2.imshow("image", image)
    img = cv2.flip(img, 1)
    top, right, bottom, left = 75, 350, 300, 590
    roi = img[top:bottom, right:left]
    roi=cv2.flip(roi,1)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    cv2.imshow('roi',gray)
    alpha=classify(gray)
    cv2.rectangle(img, (left, top), (right, bottom), (0,255,0), 2)
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,alpha,(0,130),font,5,(0,0,255),2)
    #cv2.resize(img,(1000,1000))
    cv2.imshow('img',img)
    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break;
cap.release()
cv2.destroyAllWindows()
