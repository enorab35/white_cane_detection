# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 09:38:37 2024

@author: maelb
"""

#import matplotlib.pyplot as plt
#import cv2
import os, random
import shutil

"""
image = cv2.imread("dataset/images/IMG_0329_MOV-0_jpg.rf.3c0f000617844bcd26e3d652bc80547c.jpg", 1)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

def split_dataset(directory):
    ids_img= [file_name.rsplit(".", 1)[0] for file_name in os.listdir(directory)]
    random.shuffle(ids_img)
    
    split_ratio = 0.6
    ids_train = ids_img[:int(len(ids_img)*split_ratio)]
    ids_rest = ids_img[int(len(ids_img)*split_ratio):]
    
    split_ratio = 0.5
    ids_val = ids_rest[:int(len(ids_rest)*split_ratio)]
    ids_test = ids_rest[int(len(ids_rest)*split_ratio):]
    return ids_train, ids_val, ids_test

def create_dir():
        if not os.path.exists("C:/Users/maelb/Documents/scolaire/M2/traitement d'images/seg_image/white_cane_detection/training/"):
            os.mkdir("C:/Users/maelb/Documents/scolaire/M2/traitement d'images/seg_image/white_cane_detection/training")
            os.mkdir("C:/Users/maelb/Documents/scolaire/M2/traitement d'images/seg_image/white_cane_detection/training/images")
            os.mkdir("C:/Users/maelb/Documents/scolaire/M2/traitement d'images/seg_image/white_cane_detection/training/labels")
        if not os.path.exists("C:/Users/maelb/Documents/scolaire/M2/traitement d'images/seg_image/white_cane_detection/val/"):
            os.mkdir("C:/Users/maelb/Documents/scolaire/M2/traitement d'images/seg_image/white_cane_detection/val")
            os.mkdir("C:/Users/maelb/Documents/scolaire/M2/traitement d'images/seg_image/white_cane_detection/val/images")
            os.mkdir("C:/Users/maelb/Documents/scolaire/M2/traitement d'images/seg_image/white_cane_detection/val/labels")
        if not os.path.exists("C:/Users/maelb/Documents/scolaire/M2/traitement d'images/seg_image/white_cane_detection/test"):
            os.mkdir("C:/Users/maelb/Documents/scolaire/M2/traitement d'images/seg_image/white_cane_detection/test")
            os.mkdir("C:/Users/maelb/Documents/scolaire/M2/traitement d'images/seg_image/white_cane_detection/test/images")
            os.mkdir("C:/Users/maelb/Documents/scolaire/M2/traitement d'images/seg_image/white_cane_detection/test/labels")

def create_data(directory, type_set, ids):
    lst_file = open(directory+"/"+type_set+".txt", "w")
    for img_id in ids:
        lst_file.write(directory + "/images/"+img_id + ".jpg"
        shutil.copy2( "dataset/images/"+img_id+".jpg", directory+"images/"+img_id+".jpg")
        shutil.copy2("dataset/labels/"+img_id+".txt", directory+"labels/"+img_id+".txt")
    lst_file.close()
    
def create_yaml(directory, type_set):
    # dataset.yaml
    file = open(directory + "dataset.yaml", 'w')
    file.write('train:'+ directory + type_set + '.txt\n')
    file.write('val:' + directory + type_set +'.txt\n')
    file.write('\n')
    file.write('nc: 1\n')
    file.write('names: ["white_canes_detection"]\n')
        
def main():
    img_dir = "dataset/images/"
    train, val, test = split_dataset(img_dir)
    print(len(train))
    print(len(val))
    print(len(test))
    create_dir()
    create_data("C:/Users/maelb/Documents/scolaire/M2/traitement d'images/seg_image/white_cane_detection/training/", "train", train)
    create_data("C:/Users/maelb/Documents/scolaire/M2/traitement d'images/seg_image/white_cane_detection/val/", "val", val)
    create_data("C:/Users/maelb/Documents/scolaire/M2/traitement d'images/seg_image/white_cane_detection/test/", "test", test)
    create_yaml("C:/Users/maelb/Documents/scolaire/M2/traitement d'images/seg_image/white_cane_detection/training/", "train")
main()