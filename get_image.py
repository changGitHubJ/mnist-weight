from PIL import Image
import numpy as np
import os
import struct

def load_mnist(path, kind='train'):
    """load MNIST data from designated path"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

if __name__ == '__main__':

    # training data
    X_train, y_train = load_mnist('./data/raw', kind='train')
    os.makedirs('./images', exist_ok=True)
    os.makedirs('./images/training', exist_ok=True)
    train_label = []
    for (i, x) in enumerate(X_train):
        image_mat = np.array([[x[i] for j in range(3)]+[0] for i in range(784)])
        image = image_mat.reshape(28, 28, 4)

        filename = './images/training/image_' + str(i + 1) + '.jpg'
        train_label.append(y_train[i])
        print(filename)
        pil_image = Image.fromarray(image.astype('uint8')).convert('RGB')
        pil_image.save(filename)
    np.savetxt('./images/training/label.txt', train_label)

    # evaulation data
    X_train, y_train = load_mnist('./data/raw', kind='t10k')
    os.makedirs('./images/evaluation', exist_ok=True)
    evaluate_label = []
    for (i, x) in enumerate(X_train):
        image_mat = np.array([[x[i] for j in range(3)]+[0] for i in range(784)])
        image = image_mat.reshape(28, 28, 4)

        filename = './images/evaluation/image_' + str(i + 1) + '.jpg'
        evaluate_label.append(y_train[i])
        print(filename)
        pil_image = Image.fromarray(image.astype('uint8')).convert('RGB')
        pil_image.save(filename)
    np.savetxt('./images/evaluation/label.txt', evaluate_label)