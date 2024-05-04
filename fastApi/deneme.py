import os
#pip install ultralytics==8.0.196
import torch
import time
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from IPython import display
#display.clear_output()
import json  
import ultralytics
#ultralytics.checks()
import uuid
from PIL import Image, ImageDraw

from IPython.display import display, Image

from ultralytics import YOLO
from PIL import Image
import cv2
'''
weightPATH = "ptweights/best200.pt"
imageName = "1.jpg"
imgPATH = "images/"+imageName
confidence = 0.5
model = YOLO(weightPATH)

# from PIL
im1 = cv2.imread(imgPATH)#Image.open(imgPATH)
results = model.predict(source=im1,retina_masks=True, conf=confidence, save=True) # save plotted images
#conf confidence  confidencten küçük değerlerin mask boxes değerleri olmaz!!!!! def = 0.25

result = results[0]
print(result.save_dir+'\\' + result.path)
'''
im2 = cv2.imread("runs/segment/predict4/image0.jpg")
cv2.imshow('Görüntü', im2)
cv2.waitKey(0)