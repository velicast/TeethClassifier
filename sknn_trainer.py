import numpy
import os
import sys
import time
import pickle
from skimage import data, io
from sklearn import datasets, cross_validation
from sknn.mlp import Classifier, Layer, Convolution
from sknn.platform import cpu64, threading

ORD_LABEL_YES = 1
ORD_LABEL_NO = 0

  # MODIFIQUE EL DISENIO DE LA CNN AQUI
def getCNN():
  conv_hidden_0 = Convolution(type = 'Sigmoid', 
    channels = 12, kernel_shape = (3, 3), border_mode = "full")
  poll_hidden_1 = Convolution(type = 'Sigmoid', 
    channels = 12, kernel_shape = (3, 3), pool_shape = (2, 2), pool_type = 'max')
  conv_hidden_2 = Convolution(type = 'Sigmoid', 
    channels = 8, kernel_shape = (3, 3), border_mode = "full")
  poll_hidden_3 = Convolution(type = 'Sigmoid', 
    channels = 8, kernel_shape = (3, 3), pool_shape = (2, 2), pool_type = 'max')
  relu_hidden_4 = Convolution(type = 'Rectifier', 
    channels = 8, kernel_shape = (3, 3), dropout = 0.25, border_mode = "valid")
  full_hidden_5 = Layer(type = 'Softmax')
 
  layers = [conv_hidden_0, poll_hidden_1, conv_hidden_2, poll_hidden_3, relu_hidden_4, full_hidden_5]
  cnn = Classifier(layers, n_iter = 4000, batch_size = 128, verbose = True)
  return cnn

# ----------
# Entrena la CNN y la valida con el split de datos 80% entrenamiento / 20% prueba
# La red entrenada se guarda en el archivo output_file
# ----------
def trainTeethCNN(data, labels, output_file):
  
  np_data = numpy.array(data)
  np_labels = numpy.array(labels)
  X_train, X_test, y_train, y_test = cross_validation.train_test_split(np_data, np_labels, test_size = 0.2)

  cnn = getCNN()
  cnn.fit(X_train, y_train)
  pickle.dump(cnn, open(output_file, 'wb'))
    
  print('Train Accuracy', cnn.score(X_train, y_train))
  print('Test Accuracy', cnn.score(X_test, y_test))

  # Main
  # src_path - directorio donde se encuentran las imagenes para entrenar el modelo
  # flips - incluir imagenes con flips indicados
  # rota - incluir imagenes con angulos de rotacion indicados
if __name__ == "__main__":

  src_path = sys.argv[1]
  dst_cnn_file = sys.argv[2]
  flips = sys.argv[3]
  rota = sys.argv[4]

  flip_included = {}
  rota_included = {}

  for c in flips:
    flip_included[c] = True
  for r in rota.split("_"):
    rota_included[r] = True

  images = []
  labels = []

  for file_path in os.listdir(src_path):
    target_file = src_path+"/"+file_path
    
    if os.path.isdir(target_file):
      continue

    desc = file_path.split("_")
    label, flip, rota = desc[0][0], desc[0][1], desc[1]

    if (flip not in flip_included) or (rota not in rota_included):
      continue # skip image

    label = ORD_LABEL_YES if label == "y" else ORD_LABEL_NO
    image = io.imread(target_file, as_gray = True)
    images.append(image)
    labels.append(label)

  trainTeethCNN(images, labels, dst_cnn_file)
  
