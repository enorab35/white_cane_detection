import cv2, os
import matplotlib.pyplot as plt

def disp_train(directory, num):
    plt.figure()
    plt.imshow(plt.imread(directory+num+"/results.png"))
    plt.figure()
    plt.imshow(plt.imread(directory+num+"/confusion_matrix.png"))
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(plt.imread(directory+num+"/P_curve.png"))
    plt.subplot(2, 2, 2)
    plt.imshow(plt.imread( directory+num+"/PR_curve.png"))
    plt.subplot(2, 2, 3)
    plt.imshow(plt.imread(directory+num+"/R_curve.png"))
    plt.subplot(2, 2, 4)
    plt.imshow(plt.imread( directory+num+"/F1_curve.png"))
    plt.show()


def disp_test(directory, num):
    files = os.listdir(directory+num)
    for file in files:
        plt.figure()
        plt.title(file)
        plt.imshow(plt.imread(directory+num+"/"+file))
    plt.show()
disp_train("runs/detect/train", "")
disp_test("runs/detect/val", "")

