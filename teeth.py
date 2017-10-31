import cv2
import os
import sys
import time
import teeth_util

IMG_SCALE = 1.3
FACE_QUALITY = 5
MOUTH_QUALITY = 10
MOUTH_WIDTH = 64
MOUTH_HEIGHT = 16

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

# --------------
# Extraccion de las regiones de interes en la imagen del rostro.
# Realiza el preprocesamiento de las regiones de interes
# --------------
def getMouths(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  target_face = None
  greater_area = 0
  faces = face_cascade.detectMultiScale(gray, IMG_SCALE, FACE_QUALITY)

  if len(faces) == 0:
    return []
  for (x, y, w, h) in faces:
    if greater_area < w*h:
      greater_area = w*h
      target_face = (x, y, w, h)
  
  (x, y, w, h) = target_face
  roi_gray = gray[y+(h/2) : y+h, x : x+w]
  
  result = []
  mouths = mouth_cascade.detectMultiScale(roi_gray, IMG_SCALE, MOUTH_QUALITY)

  for (mx, my,mw, mh) in mouths:
    roi_mouth = roi_gray[my : my+mh, mx : mx+mw]
    roi_mouth = cv2.equalizeHist(roi_mouth)
    roi_mouth = cv2.resize(roi_mouth, (MOUTH_WIDTH, MOUTH_HEIGHT))
    #roi_mouth = cv2.GaussianBlur(roi_mouth, (3, 3), 0)
    result.append(roi_mouth)

  return result

