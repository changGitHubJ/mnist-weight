import numpy as np
import os
import random
import tensorflow as tf
import time

# Architecture
n_hidden_1 = 256
n_hidden_2 = 256

# Parameter
learning_rate = 0.0001
training_epochs = 100
batch_size = 100
display_step = 1
TRAIN_DATA_SIZE = 10000
VALID_DATA_SIZE = 500
TEST_DATA_SIZE = 500
IMG_SIZE = 28
OUTPUT_SIZE = 10
FILTER_SIZE_1 = 32
FILTER_SIZE_2 = 64

# Batch components
trainingImages = np.zeros((TRAIN_DATA_SIZE, IMG_SIZE*IMG_SIZE + 1))
trainingLabels = np.zeros((TRAIN_DATA_SIZE, OUTPUT_SIZE + 1))
trainingWeights = np.zeros((TRAIN_DATA_SIZE, OUTPUT_SIZE + 1))
validationImages = np.zeros((VALID_DATA_SIZE, IMG_SIZE*IMG_SIZE))
validationLabels = np.zeros((VALID_DATA_SIZE, OUTPUT_SIZE))
validationWeights = np.zeros((VALID_DATA_SIZE, OUTPUT_SIZE))
testImages = np.zeros((TEST_DATA_SIZE, IMG_SIZE*IMG_SIZE))
testLabels = np.zeros((TEST_DATA_SIZE, OUTPUT_SIZE))
testWeights = np.zeros((TEST_DATA_SIZE, OUTPUT_SIZE))

def conv2d(input, weight_shape, bias_shape):
    incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2.0/incoming)**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    model = {'Weight':W, 'bias':b}
    return tf.nn.sigmoid(tf.nn.bias_add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b)), model

def max_pool(input, k=2):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def layer(input, weight_shape, bias_shape):
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    model = {'Weight':W, 'bias':b}
    return tf.nn.sigmoid(tf.matmul(input, W) + b), model

def inference(x, keep_prob):
    x = tf.reshape(x, shape=[-1, IMG_SIZE, IMG_SIZE, 1])
    with tf.variable_scope("conv_1"):
        conv_1, model_conv1 = conv2d(x, [5, 5, 1, 32], [32])
        pool_1 = max_pool(conv_1)
    with tf.variable_scope("conv_2"):
        conv_2, model_conv2 = conv2d(pool_1, [5, 5, 32, 64], [64])
        pool_2 = max_pool(conv_2)
    with tf.variable_scope("fc"):
        pool_2_flat = tf.reshape(pool_2, [-1, 7*7* 64])
        fc_1, model_fc1 = layer(pool_2_flat, [7*7*64, 1024], [1024])
        # apply dropout
        fc_1_drop = tf.nn.dropout(fc_1, keep_prob)
    with tf.variable_scope("output"):
        output, model_fc2 = layer(fc_1_drop, [1024, 10], [10])
    
    model = {'W_conv1': model_conv1['Weight'], 'b_conv1': model_conv1['bias'], \
             'W_conv2': model_conv2['Weight'], 'b_conv2': model_conv2['bias'], \
             'W_fc1': model_fc1['Weight'], 'b_fc1': model_fc1['bias'], \
             'W_fc2': model_fc2['Weight'], 'b_fc2': model_fc2['bias']}
    return output, model

def loss(output, weight):

    #xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=weight)
    #loss = tf.reduce_mean(xentropy)
    soft = tf.nn.softmax(output)
    xentropy = - tf.reduce_sum(weight * tf.log(soft), 1)
    loss = tf.reduce_mean(xentropy)
    return loss

def training(cost, global_step):
    tf.summary.scalar("cost", cost)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("validation error", (1.0 - accuracy))
    return accuracy

def write_result(output, y, model):
    return output, y, model

def openfile(filename):
    file = open(filename)
    VAL = []
    while True:
        line = file.readline()
        if(not line):
            break
        val = line.split(' ')
        VAL.append(val)
    return VAL

def read_training_data():
    fileImg = open('./data/trainImage.txt', 'r')
    for i in range(TRAIN_DATA_SIZE):
        line = fileImg.readline()
        val = line.split(',')
        trainingImages[i, :] = val
    for i in range(TRAIN_DATA_SIZE):
        for j in range(1,IMG_SIZE*IMG_SIZE + 1):
            trainingImages[i, j] /= 255.0

    filelbl = open('./data/trainLABEL.txt', 'r')
    for i in range(TRAIN_DATA_SIZE):
        line = filelbl.readline()
        val = line.split(',')
        trainingLabels[i, :] = val
    
    filewgh = open('./data/trainWEIGHT.txt', 'r')
    for i in range(TRAIN_DATA_SIZE):
        line = filewgh.readline()
        val = line.split(',')
        trainingWeights[i, :] = val

def defineBatchComtents():
    num = np.linspace(0, TRAIN_DATA_SIZE - 1, TRAIN_DATA_SIZE)
    num = num.tolist()
    COMPONENT = []
    total_batch = int(TRAIN_DATA_SIZE/batch_size)
    for i in range(total_batch):
        component = random.sample(num, batch_size)
        COMPONENT.append(component)
        for j in range(batch_size):
            cnt = 0
            while True:
                if(num[cnt] == component[j]):
                    num.pop(cnt)
                    break
                else:
                    cnt += 1
    
    return COMPONENT

def next_batch(batch_component):
    num = sorted(batch_component)
    lineNum = 0
    cnt = 0
    batch_x = []
    batch_y = []
    batch_weight = []
    while True:
        if(cnt == batch_size):
            break
        else:
            if(int(num[cnt]) == int(trainingImages[lineNum, 0])):
                image = trainingImages[lineNum, 1:IMG_SIZE*IMG_SIZE + 1]
                label = trainingLabels[lineNum, 1:OUTPUT_SIZE + 1]
                weight = trainingWeights[lineNum, 1:OUTPUT_SIZE + 1]
                batch_x.append(image)
                batch_y.append(label)
                batch_weight.append(weight)
                cnt += 1
        lineNum += 1

    return np.array(batch_x), np.array(batch_y), np.array(batch_weight)

def read_validation_data():
    fileImg = open('./data/validationImage.txt', 'r')
    for i in range(VALID_DATA_SIZE):
        line = fileImg.readline()
        val = line.split(',')
        validationImages[i, :] = val[1:IMG_SIZE*IMG_SIZE + 1]
    for i in range(VALID_DATA_SIZE):
        for j in range(IMG_SIZE*IMG_SIZE):
            validationImages[i, j] /= 255.0

    filelbl = open('./data/validationLABEL.txt', 'r')
    for i in range(VALID_DATA_SIZE):
        line = filelbl.readline()
        val = line.split(',')
        validationLabels[i, :] = val[1:OUTPUT_SIZE + 1]
    
    filewgh = open('./data/validationWEIGHT.txt', 'r')
    for i in range(VALID_DATA_SIZE):
        line = filewgh.readline()
        val = line.split(',')
        validationWeights[i, :] = val[1:OUTPUT_SIZE + 1]

def read_test_data():
    fileImg = open('./data/testImage.txt', 'r')
    for i in range(TEST_DATA_SIZE):
        line = fileImg.readline()
        val = line.split(',')
        testImages[i, :] = val[1:IMG_SIZE*IMG_SIZE + 1]
    for i in range(TEST_DATA_SIZE):
        for j in range(IMG_SIZE*IMG_SIZE):
            testImages[i, j] /= 255.0
    
    filelbl = open('./data/testLABEL.txt', 'r')
    for i in range(TEST_DATA_SIZE):
        line = filelbl.readline()
        val = line.split(',')
        testLabels[i, :] = val[1:OUTPUT_SIZE + 1]

    filewgh = open('./data/testWEIGHT.txt', 'r')
    for i in range(TEST_DATA_SIZE):
        line = filewgh.readline()
        val = line.split(',')
        testWeights[i, :] = val[1:OUTPUT_SIZE + 1]

def write_model(W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2, output, label):
    output_np = np.zeros([TEST_DATA_SIZE, OUTPUT_SIZE])
    for k in range(TEST_DATA_SIZE): 
        for j in range(OUTPUT_SIZE):
            output_np[k, j] = output[k, j]
    np.savetxt('./model/output.txt', output_np)

    for k in range(TEST_DATA_SIZE): 
        for j in range(OUTPUT_SIZE):
            output_np[k, j] = label[k, j]
    np.savetxt('./model/testLabel.txt', output_np)

    W_conv1_np = np.zeros(5*5*1*FILTER_SIZE_1)
    for i in range(5):
        for j in range(5):
            for k in range(1):
                for l in range(FILTER_SIZE_1):
                    W_conv1_np[i*5 + j + 25*l] = W_conv1[i,j,k,l]
    np.savetxt('./model/W_conv1.txt', W_conv1_np)

    b_conv1_np = np.zeros(FILTER_SIZE_1)
    for i in range(FILTER_SIZE_1):
        b_conv1_np[i] = b_conv1[i]
    np.savetxt('./model/b_conv1.txt', b_conv1_np)

    W_conv2_np = np.zeros(5*5*FILTER_SIZE_1*FILTER_SIZE_2)
    for i in range(5):
        for j in range(5):
            for k in range(FILTER_SIZE_1):
                for l in range(64):
                    W_conv2_np[i*5 + j + 25*k + 25*FILTER_SIZE_1*l] = W_conv2[i,j,k,l]
    np.savetxt('./model/W_conv2.txt', W_conv2_np)

    b_conv2_np = np.zeros(FILTER_SIZE_2)
    for i in range(FILTER_SIZE_2):
        b_conv2_np[i] = b_conv2[i]
    np.savetxt('./model/b_conv2.txt', b_conv2_np)

    W_fc1_np = np.zeros(7*7*FILTER_SIZE_2*1024)
    for i in range(7*7*FILTER_SIZE_2):
        for j in range(1024):
            W_fc1_np[i * 1024 + j] = W_fc1[i, j]
    np.savetxt('./model/W_fc1.txt', W_fc1_np)

    b_fc1_np = np.zeros(1024)
    for i in range(1024):
        b_fc1_np[i] = b_fc1[i]
    np.savetxt('./model/b_fc1.txt', b_fc1_np)
    
    W_fc2_np = np.zeros(1024 * OUTPUT_SIZE)
    for i in range(1024):
        for j in range(OUTPUT_SIZE):
            W_fc2_np[i * OUTPUT_SIZE + j] = W_fc2[i, j]
    np.savetxt('./model/W_fc2.txt', W_fc2_np)

    b_fc2_np = np.zeros(OUTPUT_SIZE)
    for i in range(OUTPUT_SIZE):
        b_fc2_np[i] = b_fc2[i]
    np.savetxt('./model/b_fc2.txt', b_fc2_np)

def write_log(msg):
    os.makedirs('./log', exist_ok=True)
    file = open('./log/trainingLog.txt', mode='a')
    file.write(msg + '\n')
    file.close()

if __name__=='__main__':    
    with tf.device("/gpu:0"):
        with tf.Graph().as_default():
            with tf.variable_scope("scope_model"):
                x = tf.placeholder("float", [None, IMG_SIZE*IMG_SIZE])
                #y = tf.placeholder("float", [None, OUTPUT_SIZE])
                weight = tf.placeholder("float", [None, OUTPUT_SIZE])
                keep_prob = tf.placeholder(tf.float32)

                read_training_data()
                read_validation_data()
                read_test_data()
                
                output, model = inference(x, keep_prob)
                cost = loss(output, weight)
                global_step = tf.Variable(0, name='global_step', trainable=False)
                train_op = training(cost, global_step)
                eval_op = evaluate(output, weight)
                test_op = write_result(output, weight, model)
                #summary_op = tf.summary.merge_all()
                #saver = tf.train.Saver()
                sess = tf.Session()
                #summary_writer = tf.summary.FileWriter("conv_mnist_logs/", graph=sess.graph)
                init_op = tf.global_variables_initializer()
                sess.run(init_op)

                # Training cycle
                for epoch in range(training_epochs):
                    avg_cost = 0.
                    total_batch = int(TRAIN_DATA_SIZE/batch_size)
                    batch_component = defineBatchComtents()
                    # loop over all batchs
                    for i in range(total_batch):
                        minibatch_x, minibatch_y, minibatch_weight = next_batch(batch_component[i])
                        sess.run(train_op, feed_dict={x: minibatch_x, weight: minibatch_weight, keep_prob: 0.5})
                        avg_cost += sess.run(cost, feed_dict={x: minibatch_x, weight: minibatch_weight, keep_prob: 0.5})/total_batch

                    # display logs per step
                    if epoch % display_step == 0:
                        accuracy = sess.run(eval_op, feed_dict={x: validationImages, weight: validationWeights, keep_prob: 0.5})
                        msg = "Epoch: " + str(epoch+1) + ", cost = " + "{:.9f}".format(avg_cost) + ", Validation Error = " + "{:.9f}".format(1 - accuracy)
                        print(msg)
                        write_log(msg)
                        #summary_str = sess.run(summary_op, feed_dict={x: minibatch_x, y: minibatch_y, weight: minibatch_weight, keep_prob: 0.5})
                        #summary_writer.add_summary(summary_str, sess.run(global_step))
                        #saver.save(sess, "conv_mnist_logs/model_checkpoint", global_step=global_step)

                print("Optimizer finished!")
                accuracy = sess.run(eval_op, feed_dict={x: testImages, weight: testWeights, keep_prob: 1})
                out, lbl, mdl = sess.run(test_op, feed_dict={x: testImages, weight: testWeights, keep_prob: 1})
                msg = "Epoch: " + str(-1) + ", cost = " + "{:.9f}".format(-1) + ", Test Accuracy = " + "{:.9f}".format(accuracy)
                print(msg)
                write_log(msg)

                # output model
                os.makedirs('./model', exist_ok=True)
                for f in os.listdir('./model/'):
                    os.remove('./model/'+f)

                #tmp = 
                write_model(mdl['W_conv1'], mdl['b_conv1'], mdl['W_conv2'], mdl['b_conv2'], mdl['W_fc1'], mdl['b_fc1'], mdl['W_fc2'], mdl['b_fc2'], out, lbl)
                
                print('Saved a model.')

                sess.close()
                

