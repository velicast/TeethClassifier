import cv2
import os

def ensure_dir(path):

  if not os.path.exists(path):
    os.makedirs(path)

def saveJPEG(path, img, quality = 100):
  cv2.imwrite(path+".jpeg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

def rotateImage(img, angle):

  h, w = img.shape
  M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
  return cv2.warpAffine(img, M, (w, h))
