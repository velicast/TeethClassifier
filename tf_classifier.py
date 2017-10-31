import os
import sys
import cv2
import shutil
import tensorflow as tf
import teeth
import teeth_util

TMP_FILE_NAME = "./tmpHZQ12Djh27XVBdl_Xxas"

# ---------
# Cargar red neuronal entrenada
# ---------
graph_def = None

with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
  tf.import_graph_def(graph_def, name = '')

# ---------
# Clasifica la imagen correspondiente a region de interes, 
# con su respectivo valor de confidencia (probabilidad de pertenencia)
# ---------
def classifyMouth(image_path):

  image_data = tf.gfile.FastGFile(image_path, 'rb').read()
  label_lines = [line.rstrip() for line 
                     in tf.gfile.GFile("retrained_labels.txt")]

  with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    label = label_lines[top_k[0]]
    score = predictions[0][top_k[0]]
    return (label, score)

# ---------
# Clasifica la imagen del rostro
# ---------
def classify(face_image_path):

  img = cv2.imread(face_image_path)
  mouths = teeth.getMouths(img)
  best_score = -1
  best_label = ""

  for mouth in mouths:
    teeth_util.saveJPEG(TMP_FILE_NAME, mouth)
    (label, score) = classifyMouth(TMP_FILE_NAME+".jpeg")

    if score >= best_score:
      best_score = score
      best_label = label
  
  os.remove(TMP_FILE_NAME)
  return best_label

  # Main
  #   src_dir = directorio donde se encuentran las imagenes a clasificar
  #   dst_dir = directorio donde se almacenaran las imagenes clasificadas
  #   Cada categoria creara un subdirectorio en dst_dir/
if __name__ == "__main__":
  
  src_dir = sys.argv[1]
  dst_dir = sys.argv[2]
  teeth_util.ensure_dir(dst_dir)

  for file_path in os.listdir(src_dir):
    target_file = src_dir+"/"+file_path

    if os.path.isdir(target_file):
      continue

    label = classify(target_file)
    teeth_util.ensure_dir(dst_dir+"/"+label)
    shutil.copy(target_file, dst_dir+"/"+label+"/"+file_path)

