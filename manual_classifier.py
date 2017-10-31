import cv2
import os
import sys
import time
import teeth_util
import teeth

# ------------------------------------------------------------------------
# CODIGO UTILITARIO PARA REALIZAR LA CLASIFICACION MANUAL DE LAS IMAGENES 
# ------------------------------------------------------------------------

ROTATE_ANGLES = [0, 5, 10]

def manualLabeling(image_file, output_dir):

  img = cv2.imread(image_file)
  label_dist = {}

  for mouth in teeth.getMouths(img):
    cv2.imshow('img', mouth)
    label = cv2.waitKey(0)&0xFF

    if label == 27: # Finish ESC
      cv2.destroyAllWindows()
      return (label_dist, False)
    cv2.destroyAllWindows()

    label = chr(label)

    if label == 'b': # skip image
      continue

    if label not in label_dist:
      label_dist[label] = 0
    label_dist[label] += 1

    base_dir = output_dir+"/"+label
    base_name = base_dir+"/"+label
    teeth_util.ensure_dir(base_dir)

    for angle in ROTATE_ANGLES:
      img = teeth_util.rotateImage(mouth, angle)
      angle = str(angle)

      img_name = base_name+"o_"+angle+"_"+str(time.time())
      teeth_util.saveJPEG(img_name, img)
      
      img_name = base_name+"v_"+angle+"_"+str(time.time())
      vertical_flip = cv2.flip(img, 1)
      teeth_util.saveJPEG(img_name, vertical_flip)

      img_name = base_name+"h_"+angle+"_"+str(time.time())
      horizontal_flip = cv2.flip(img, 0)
      teeth_util.saveJPEG(img_name, horizontal_flip)
      
      img_name = base_name+"b_"+angle+"_"+str(time.time())
      both_flip = cv2.flip(img, -1)
      teeth_util.saveJPEG(img_name, both_flip)
  
  return (label_dist, True)

  # Main
  #   src_dir = directorio donde se encuentran las imagenes a clasificar
  #   dst_dir = directorio donde se almacenaran las imagenes clasificadas
  #   Cada categoria creara un subdirectorio en dst_dir/
if __name__ == "__main__":

  src_dir = sys.argv[1]
  dst_dir = sys.argv[2]
  label_dist = {}

  teeth_util.ensure_dir(dst_dir)

  for file_path in os.listdir(src_dir):
    target_file = src_dir+"/"+file_path
    
    if os.path.isdir(target_file):
      continue

    l_dist, next_img = manualLabeling(target_file, dst_dir)
    
    for (l, f) in l_dist.iteritems():
      if l not in label_dist:
        label_dist[l] = 0
      label_dist[l] += f

    if not next_img:
      break

  print "LABELS DISTRIBUTION"

  for (l, f) in label_dist.iteritems():
    print "Label: ", l, " Frequency: ", f

