# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 09:38:37 2024

@author: maelb
"""
import os, random
import shutil


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
        if not os.path.exists("training/"):
            os.mkdir("training")
            os.mkdir("training/images")
            os.mkdir("training/labels")
        if not os.path.exists("test"):
            os.mkdir("test")
            os.mkdir("test/images")
            os.mkdir("test/labels")


def create_data(directory, type_set, ids):
    lst_file = open(directory+type_set+".txt", "w")
    i=0
    for img_id in ids:
        lst_file.write(directory + "images/"+type_set+str(i) + ".jpg\n")
        shutil.copy2( "dataset/images/"+img_id+".jpg", directory+"images/"+type_set+str(i)+".jpg")
        shutil.copy2("dataset/labels/"+img_id+".txt", directory+"labels/"+type_set+str(i)+".txt")
        i+=1
    lst_file.close()


def filtre_box(directory):
    files = os.listdir(directory)
    for file in files:
        lines = []
        path = os.path.join(directory, file)
        f = open(path, "r")
        lines_f = f.readlines()
        for line in lines_f:
            valeurs = [valeur for valeur in line.strip().split()]
            if len(valeurs) >=5:
                valeurs = valeurs[:5]
            lines.append(valeurs)
        f.close()
        f = open(path, "w")
        for val in lines:
            val_up = ' '.join(map(str, val)) + '\n'
            f.write(val_up)
        f.close()


def create_yaml():
    # dataset.yaml
    file = open("training/dataset.yaml", 'w')
    file.write('train: train.txt\n')
    file.write('val: val.txt\n')
    file.write('\n')
    file.write('nc: 2\n')
    file.write('names: ["umbrella", "white_canes"]\n')
    file.close()
    file = open("test/dataset.yaml", "w")
    file.write('train: train.txt\n')
    file.write('val: test.txt\n')
    file.write('\n')
    file.write('nc: 2\n')
    file.write('names: ["umbrella", "white_canes"]\n')
    file.close()


def main():
    classes = ["0", "1"]
    img_dir = "dataset/images/"
    train, val, test = split_dataset(img_dir)
    print(len(train))
    print(len(val))
    print(len(test))
    create_dir()
    create_data("training/", "train", train)
    create_data("training/", "val", val)
    create_data("test/", "test", test)
    filtre_box("training/labels")
    create_yaml()
main()