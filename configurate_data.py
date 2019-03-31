import math
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from PIL import Image

TRAIN_DATA_SIZE = 10000
VALID_DATA_SIZE = 500
TEST_DATA_SIZE = 500
IMG_SIZE = 28
OUTPUT_SIZE = 10
DEFECT = 0.8
NO_DEFECT = 0.2


if __name__ == "__main__":

    init = tf.global_variables_initializer()
    sess = tf.Session()
    with sess.as_default():

        if not os.path.exists('./data'):
            os.mkdir('./data')

        # remove old file
        if(os.path.exists('./data/trainImage.txt')):
           os.remove('./data/trainImage.txt')
        if(os.path.exists('./data/validationImage.txt')):
           os.remove('./data/validationImage.txt')
        if(os.path.exists('./data/testImage.txt')):
           os.remove('./data/testImage.txt')
        if(os.path.exists('./data/trainLABEL.txt')):
            os.remove('./data/trainLABEL.txt')
        if(os.path.exists('./data/trainWEIGHT.txt')):
            os.remove('./data/trainWEIGHT.txt')
        if(os.path.exists('./data/validationLABEL.txt')):
            os.remove('./data/validationLABEL.txt')
        if(os.path.exists('./data/validationWEIGHT.txt')):
            os.remove('./data/validationWEIGHT.txt')
        if(os.path.exists('./data/testLABEL.txt')):
            os.remove('./data/testLABEL.txt')
        if(os.path.exists('./data/testWEIGHT.txt')):
            os.remove('./data/testWEIGHT.txt')

        # load training images (1-10000)        
        for k in range(TRAIN_DATA_SIZE):
            filename = './images/training/image_' + str(k + 1) + '.jpg'
            print(filename)
            imgtf = tf.read_file(filename)
            img = tf.image.decode_jpeg(imgtf, channels=1)
            array = img.eval()
            line = str(k)
            for i in range(IMG_SIZE):
                for j in range(IMG_SIZE):
                    line = line + ',' + str(array[i, j, 0])
            line = line + '\n'
            file = open('./data/trainImage.txt', 'a')
            file.write(line)
            file.close()

        # load validation images (1-1000)
        evaluate_label = np.loadtxt('./images/evaluation/label.txt')
        for k in range(VALID_DATA_SIZE + TEST_DATA_SIZE):
            filename = './images/evaluation/image_' + str(k + 1) + '.jpg'
            print(filename)
            imgtf = tf.read_file(filename)
            img = tf.image.decode_jpeg(imgtf, channels=1)
            array = img.eval()
            line = str(k)
            for i in range(IMG_SIZE):
                for j in range(IMG_SIZE):
                    line = line + ',' + str(array[i, j, 0])
            line = line + '\n'
            if(k < VALID_DATA_SIZE):
                file = open('./data/validationImage.txt', 'a')
                file.write(line)
                file.close()
            else:
                file = open('./data/testImage.txt', 'a')
                file.write(line)
                file.close()

        # label #
        train_label = np.loadtxt('./images/training/label.txt')
        trnLABEL = []
        trnWEIGHT = []
        for k in range(TRAIN_DATA_SIZE):
            label = np.zeros([OUTPUT_SIZE + 1])
            label[0] = k
            label[1 + int(train_label[k])] = 1
            weight = np.zeros([OUTPUT_SIZE + 1])
            weight[0] = k
            weight[1:OUTPUT_SIZE + 1] = NO_DEFECT
            weight[1 + int(train_label[k])] = DEFECT
            trnLABEL.append(label)
            trnWEIGHT.append(weight)

        valid_label = np.loadtxt('./images/evaluation/label.txt')
        valLABEL = []
        valWEIGHT = []
        tstLABEL = []
        tstWEIGHT = []
        for k in range(VALID_DATA_SIZE + TEST_DATA_SIZE):
            label = np.zeros([OUTPUT_SIZE + 1])
            label[0] = k
            label[1 + int(valid_label[k])] = 1
            weight = np.zeros([OUTPUT_SIZE + 1])
            weight[0] = k
            weight[1:OUTPUT_SIZE + 1] = NO_DEFECT
            weight[1 + int(valid_label[k])] = DEFECT

            if(k < VALID_DATA_SIZE):
                valLABEL.append(label)
                valWEIGHT.append(weight)
            else:
                tstLABEL.append(label)
                tstWEIGHT.append(weight)

        # normalize
        w_array = np.array(trnWEIGHT)
        for k in range(TRAIN_DATA_SIZE):
            s = sum(w_array[k, 1:OUTPUT_SIZE + 1])
            w_array[k, 1:OUTPUT_SIZE + 1] = w_array[k, 1:OUTPUT_SIZE + 1]/s
        trnWEIGHT = w_array.tolist()
        
        w_val_array = np.array(valWEIGHT)
        for k in range(VALID_DATA_SIZE):
            s = sum(w_val_array[k, 1:OUTPUT_SIZE + 1])
            w_val_array[k, 1:OUTPUT_SIZE + 1] = w_val_array[k, 1:OUTPUT_SIZE + 1]/s
        valWEIGHT = w_val_array.tolist()

        w_tst_array = np.array(tstWEIGHT)
        for k in range(TEST_DATA_SIZE):
            s = sum(w_tst_array[k, 1:OUTPUT_SIZE + 1])
            w_tst_array[k, 1:OUTPUT_SIZE + 1] = w_tst_array[k, 1:OUTPUT_SIZE + 1]/s
        tstWEIGHT = w_tst_array.tolist()
        
        np.savetxt('./data/trainLABEL.txt', trnLABEL, fmt='%d', delimiter=',')
        np.savetxt('./data/trainWEIGHT.txt', trnWEIGHT, fmt='%.10f', delimiter=',')
        np.savetxt('./data/validationLABEL.txt', valLABEL, fmt='%d', delimiter=',')
        np.savetxt('./data/validationWEIGHT.txt', valWEIGHT, fmt='%.10f', delimiter=',')
        np.savetxt('./data/testLABEL.txt', tstLABEL, fmt='%d', delimiter=',')
        np.savetxt('./data/testWEIGHT.txt', tstWEIGHT, fmt='%.10f', delimiter=',')

        sess.close()
