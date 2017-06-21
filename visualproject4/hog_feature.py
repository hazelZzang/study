"""
Visual Computing Project 3.
find the HOG feature vector for image
입력 : image
출력 : HOG feature vector
"""
import tensorflow as tf
import numpy as np
import os
import cv2
import csv
##-------- 파일 위치
pos_train_file = './train/pos_train'
neg_train_file = './train/neg_train'
pos_test_file = './test/pos_test'
neg_test_file = './test/neg_test'


#----------- hog feature vector를 구한다.
def get_hog_feature(img):
    b_size = 16
    b_overlap = 1/2
    b_shift = int(b_size * b_overlap)
    degree = 20

    #이미지를 흑백으로 읽어들인다.
    img = img.astype(np.float32)


    #sample size : 134, 70
    rows, cols = img.shape
    Ix = img.copy()
    Iy = img.copy()


    #gradient 계산
    for i in range(1, cols-2):
        Ix[:,i+1] = img[:,i]-img[:,i+2]

    for i in range(1, rows-2):
        Iy[i+1,:] = img[i,:]-img[i+2,:]


    #orientation
    angle = img.copy()
    angle = angle.astype(np.float32)

    magnitude = img.copy()
    magnitude = magnitude.astype(np.float32)

    for i in range(rows):
        for j in range(cols):
            if(Ix[i,j] == 0):
                angle[i][j] = 0
            else:
                angle[i][j] = np.degrees(np.arctan(Iy[i,j])/Ix[i,j]) + 90
                magnitude[i][j] = np.sqrt(Ix[i,j]**2 + Iy[i,j]**2)
    block = []

    #하나의 block의 angle과 magnitude 값을 ang_pat, mag_pat에 저장한다.
    for i in range(int(rows/b_shift)):
        for j in range(int(cols/b_shift)):

            #이미지 파일이 b_size의 배수가 아니라서
            #마지막 끝이 맞지 않는 경우
            index_i = b_shift*i
            index_j = b_shift*j
            if(index_i+b_size >= rows and index_j+b_size >= cols):
                mag_pat = magnitude[rows-b_size : rows , cols-b_size:cols]
                ang_pat = angle[rows-b_size : rows , cols-b_size:cols]
            elif(index_i+b_size >= rows):
                mag_pat = magnitude[rows - b_size : rows, index_j : index_j+b_size]
                ang_pat = angle[rows - b_size : rows, index_j : index_j+b_size]
            elif(index_j+b_size >= cols):
                mag_pat = magnitude[index_i : index_i+b_size, cols - b_size:cols]
                ang_pat = angle[index_i : index_i+b_size, cols - b_size:cols]
            else:
                mag_pat = magnitude[index_i : index_i+b_size, index_j: index_j+b_size]
                ang_pat = angle[index_i : index_i+b_size, index_j: index_j+b_size]

            #하나의 block에서 각 셀의 magnitude와 angle 값을 mag, ang 에 저장한다.
            for r in range(2):
                for c in range(2):
                    index_r = b_shift*r
                    index_c = b_shift*c
                    mag = mag_pat[index_r : index_r +b_shift , index_c : index_c +b_shift]
                    ang = ang_pat[index_r : index_r +b_shift , index_c : index_c +b_shift]

                    #histogram을 제작한다.
                    #기여도를 나누어서 배분한다.
                    hist = np.zeros(int(180/degree))
                    for n_r in range(b_shift):
                        for n_c in range(b_shift):
                            ang_cell = ang[n_r,n_c]
                            ang_num = int(ang[n_r,n_c]/degree)
                            if(ang_num == 0):
                                hist[1] = hist[1] + mag[n_r, n_c] * ((1) * degree - ang_cell) / degree
                                hist[8] = hist[8] + mag[n_r, n_c] * (ang_cell - 8 * degree) / degree
                            elif(ang_num >= 8):
                                hist[8] = hist[8] + mag[n_r, n_c] * ((8) * degree - ang_cell) / degree
                                hist[1] = hist[1] + mag[n_r, n_c] * (ang_cell - 1 * degree) / degree
                            else:
                                hist[ang_num] = hist[ang_num] + mag[n_r,n_c]*((ang_num+1)*degree - ang_cell)/degree
                                hist[ang_num + 1] = hist[ang_num + 1] + mag[n_r,n_c]*(ang_cell - ang_num*degree)/degree

                    #normalize using L1-Norm
                    hist = hist/np.sqrt(np.linalg.norm(hist)**2+.01)
                    block = np.append(block,hist)

    #normalize using L2-Norm
    block = block/np.sqrt(np.linalg.norm(block)**2+.001)*1000
    return np.array(block)

#------ 이미지 파일을 흑백으로 읽어온다
def read_file(pos_file, neg_file):
    with open('./feature_vector.csv', 'w') as f:
        data = []
        label = []
        for p in os.listdir(pos_file):
            img = cv2.imread(pos_file+'/'+p ,0 )
            data.append(get_hog_feature(img))

            label.append([0.0,1.0])

        for n in os.listdir(neg_file):
            img = cv2.imread(neg_file+'/'+n ,0)
            resize_img = cv2.resize(img, (70,134))
            vec = get_hog_feature(resize_img)
            data.append(vec)

            label.append([1.0, 0.0])

    return np.array(data), np.array(label)
#read_file(pos_train_file, neg_train_file)


##------- 하이퍼 파라미터
batch_size = 5
data_size = 1200
image_size_r = 96
image_size_c = 48
input_size = image_size_r * image_size_c
target_size = 2
filter_size = 5
dep1 = 32
dep2 = 64
dep3 = 128
fc_dep1 = 1024


x_input = tf.placeholder(tf.float32, [batch_size , None])
y_input = tf.placeholder(tf.float32, [None, target_size])

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


#input [batch, height, width, channels]
#stride : 이동 칸 수, padding : 입력과 출력이 동일하도록,
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='SAME')

#구간을 나눠서 더하면 linear하지 않다.
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding='SAME')

initializer = tf.contrib.layers.xavier_initializer()

x_image = tf.reshape(x_input , [-1, image_size_r, image_size_c, 1])

#------------ 1층
W_conv1 = tf.get_variable(name="weight_conv1",initializer=initializer, shape=[filter_size, filter_size, 1, dep1])
b_conv1 = bias_variable([dep1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


#------------- 2층
W_conv2 = tf.get_variable(name="weight_conv2",initializer=initializer, shape=[filter_size, filter_size, dep1, dep2])
b_conv2 = bias_variable([dep2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#------------- 3층
W_conv3 = tf.get_variable(name="weight_conv3",initializer=initializer, shape=[filter_size, filter_size, dep2, dep3])
b_conv3 = bias_variable([dep3])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)
fc1 = 12*6*dep3


#------------------ fully connected
W_fc1 = tf.get_variable(name="weight_fully_connected1",initializer=initializer, shape=[fc1, fc_dep1])
b_fc1 = bias_variable([fc_dep1])

h_pool3_flat = tf.reshape(h_pool3, [-1, fc1])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = tf.get_variable(name="weight_fully_connected2",initializer=initializer, shape=[fc_dep1, target_size])
b_fc2 = bias_variable([target_size])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#----------------- 손실함수
cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#---------------- 정확도 계산
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_input, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



#------ 학습 데이터
data, label = read_file(pos_train_file, neg_train_file)
batch_images, batch_labels = tf.train.shuffle_batch([data, label],
                                                    batch_size=batch_size,
                                                    capacity=2000,
                                                 min_after_dequeue=1000,
                                                    enqueue_many=True)


with tf.Session() as sess:
    #-------- 배치큐를 위한 스레드 생성
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    sess.run(tf.global_variables_initializer())

    for epoch in range(1000):
        for i in range(int(data_size/batch_size) + 1):
            b_i, b_l = sess.run([batch_images, batch_labels])
            train_step.run(feed_dict = {x_input:b_i, y_input:b_l, keep_prob :0.5})
        if epoch % 100 == 0:
            print(epoch)
        count = 0
        all_accuracy = 0
    """
    for i in range(1000):
        count += 1

        t_i, t_l = sess.run([t_batch_images, t_batch_labels])
        train_accuracy = accuracy.eval(feed_dict={x_input: t_i, y_input: t_l, keep_prob: 1.0})
        all_accuracy += train_accuracy
        print(train_accuracy)
    print(all_accuracy/count)
    """
    coord.request_stop()
    coord.join(threads)

