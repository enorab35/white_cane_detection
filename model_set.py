# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 09:38:37 2024

@author: maelb
"""

#import matplotlib.pyplot as plt
import cv2
import os, random

"""
image = cv2.imread("dataset/images/IMG_0329_MOV-0_jpg.rf.3c0f000617844bcd26e3d652bc80547c.jpg", 1)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

def split_dataset(directory):
    ids_img= [file_name.split(".")[0] for file_name in os.listdir(directory)]
    random.shuffle(ids_img)
    
    split_ratio = 0.6
    ids_train = ids_img[:int(len(ids_img)*split_ratio)]
    ids_rest = ids_img[int(len(ids_img)*split_ratio):]
    
    split_ratio = 0.5
    ids_val = ids_rest[:int(len(ids_rest)*split_ratio)]
    ids_test = ids_rest[int(len(ids_rest)*split_ratio):]
    return ids_train, ids_val, ids_test

def main():
    img_dir = "../images/"
    train, val, test = split_dataset(img_dir)
    print(len(train))
    print(len(val))
    print(len(test))
    
main()