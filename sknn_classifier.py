import numpy
import cv2
import os
import sys
import time
import shutil
import teeth_util
import pickle
import teeth
import sknn_trainer

IDX_LABEL_NO = 0

# ---------
# Clasifica la imagen correspondiente a region de interes, 
# con su respectivo valor de confidencia (probabilidad de pertenencia)
# ---------
def classifyMouth(cnn, mouth):

  np_array = numpy.array([mouth])
  prob = cnn.predict_proba(np_array)
  prob = prob[0]
  
  if prob[IDX_LABEL_NO] > prob[1-IDX_LABEL_NO]:
    return (sknn_trainer.ORD_LABEL_NO, prob[IDX_LABEL_NO])

  return (sknn_trainer.ORD_LABEL_YES, prob[1-IDX_LABEL_NO])

# ---------
# Clasifica la imagen del rostro
# ---------
def classify(cnn, image_path):

  img = cv2.imread(image_path)
  mouths = teeth.getMouths(img)
  best_score = -1
  best_label = -1
  
  for mouth in mouths:
    (label, score) = classifyMouth(cnn, mouth)

    if score >= best_score:
      best_score = score
      best_label = label
  
  return "y" if best_label == sknn_trainer.ORD_LABEL_YES else "n"

  # Main
  #   src_dir = directorio donde se encuentran las imagenes a clasificar
  #   dst_dir = directorio donde se almacenaran las imagenes clasificadas
  #   cnn_file = archivo que contiene la red neuronal entrenada
  #   Cada categoria creara un subdirectorio en dst_dir/
if __name__ == "__main__":

  src_dir = sys.argv[1]
  dst_dir = sys.argv[2]
  cnn_file = sys.argv[3]
  teeth_util.ensure_dir(dst_dir)

  cnn = pickle.load(open(cnn_file, 'rb'))
  IDX_LABEL_NO = 1

  if cnn.classes_[0][0] == sknn_trainer.ORD_LABEL_NO:
    IDX_LABEL_NO = 0
    
  for file_path in os.listdir(src_dir):
    target_file = src_dir+"/"+file_path

    if os.path.isdir(target_file):
      continue

    label = classify(cnn, target_file)
    teeth_util.ensure_dir(dst_dir+"/"+label)
    shutil.copy(target_file, dst_dir+"/"+label+"/"+file_path)

