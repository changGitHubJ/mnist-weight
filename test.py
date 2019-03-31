import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

NUM = 50
IMG_SIZE = 28
OUTPUT_SIZE = 10

def readImages(filename):
    images = np.zeros((NUM, IMG_SIZE*IMG_SIZE))
    fileImg = open(filename)
    for k in range(NUM):
        line = fileImg.readline()
        if(not line):
            break
        val = line.split(',')
        for i in range(IMG_SIZE*IMG_SIZE):
            images[k, i] = float(val[i + 1])
    return images

def readLabels(filename):
    labels = np.zeros((NUM, OUTPUT_SIZE))
    fileImg = open(filename)
    for k in range(NUM):
        line = fileImg.readline()
        if(not line):
            break
        val = line.split(',')
        for i in range(OUTPUT_SIZE):
            labels[k, i] = float(val[i + 1])
    return labels

if __name__=='__main__':
    trn_image = readImages('./data/trainImage.txt')
    trn_label = readLabels('./data/trainWEIGHT.txt')
    val_image = readImages('./data/validationImage.txt')
    val_label = readLabels('./data/validationWEIGHT.txt')
    tst_image = readImages('./data/testImage.txt')
    tst_label = readLabels('./data/testWEIGHT.txt')
    output = np.loadtxt('./model/output.txt')

    for i in range(NUM):
        plt.figure(figsize=(5, 5))
        plt.subplot(3, 3, 1)
        fig = plt.imshow(trn_image[i, :].reshape([IMG_SIZE, IMG_SIZE]), vmin=0, vmax=255, cmap='jet', aspect='auto')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.colorbar()
        plt.subplot(3, 3, 2)
        fig = plt.imshow(trn_label[i, :].reshape([OUTPUT_SIZE, 1]), cmap='jet', aspect='auto')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.colorbar()

        plt.subplot(3, 3, 4)
        fig = plt.imshow(val_image[i, :].reshape([IMG_SIZE, IMG_SIZE]), vmin=0, vmax=255, cmap='jet', aspect='auto')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.colorbar()
        plt.subplot(3, 3, 5)
        fig = plt.imshow(val_label[i, :].reshape([OUTPUT_SIZE, 1]), cmap='jet', aspect='auto')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.colorbar()

        plt.subplot(3, 3, 7)
        fig = plt.imshow(tst_image[i, :].reshape([IMG_SIZE, IMG_SIZE]), vmin=0, vmax=255, cmap='jet', aspect='auto')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.colorbar()
        plt.subplot(3, 3, 8)
        fig = plt.imshow(tst_label[i, :].reshape([OUTPUT_SIZE, 1]), cmap='jet', aspect='auto')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.colorbar()
        plt.subplot(3, 3, 9)
        fig = plt.imshow(output[i, :].reshape([OUTPUT_SIZE, 1]), cmap='jet', aspect='auto')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.colorbar()
        plt.show()
