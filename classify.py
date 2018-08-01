from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")

COMPRESSION = 2
IMAGE_DIMS = (int(2516/COMPRESSION),int(3272/COMPRESSION),1)

args = vars(ap.parse_args())
image = cv2.imread(args["image"],0)
output = image.copy()
image = cv2.resize(image, ( IMAGE_DIMS[1],IMAGE_DIMS[0]))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

model = load_model(args["model"])
proba = model.predict(image)[0]
name="test"
proba=proba*255
proba = proba.astype(np.uint8)

#ret,thresh = cv2.threshold(proba,0.1,1,cv2.THRESH_BINARY_INV)
#ret3,th3 = cv2.threshold(proba,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
im_thresh2= cv2.adaptiveThreshold(proba,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, -5)


#cv2.imwrite(name+".bmp",thresh)
cv2.imshow('binarized',im_thresh2)
cv2.waitKey(0)
cv2.destroyAllWindows()