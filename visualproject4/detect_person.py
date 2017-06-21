"""
visual computing project
20143051
kimhyeji

이미지에서 사람을 검출하는 프로그램
test1.png 형태의 test 파일이 주어진 경우
사람을 검출해 직사각형으로 표시해준다.

에폭 20, 0.94
에폭 500, 0.95
"""


import pyramidCropping as pc
import tensorflow as tf
import numpy as np
import os
import cv2

##-------- 파일 위치
pos_train_file = './train/pos_train'
neg_train_file = './train/neg_train'
pos_test_file = './test/pos_test'
neg_test_file = './test/neg_test'
save_dir = './checkpoint_dir'

##------- 하이퍼 파라미터

#배치 사이즈
batch_size = 5
#데이터 갯수
data_size = 1200
#이미지 크기(row, col)
image_size_r = 134
image_size_c = 70
input_size = image_size_r * image_size_c
#목표값(one_hot_encoding)
target_size = 2
#합성곱에서의 필터 사이즈
filter_size = 5
#층별 깊이
dep1 = 32
dep2 = 64
dep3 = 128
fc_dep1 = 1024
#test data 갯수
test_data_num = 4


x_input = tf.placeholder(tf.float32, [None, image_size_r, image_size_c])
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
fc1 = 17*9*dep3

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
result = tf.argmax(y_conv, 1)
correct_prediction = tf.equal(result, tf.argmax(y_input, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#------ 훈련용 데이터
#------ 이미지 파일을 흑백으로 읽어온다
def read_file(pos_file, neg_file):
    data = []
    label = []
    for p in os.listdir(pos_file):
        img = cv2.imread(pos_file+'/'+p ,0 )
        data.append(img/255.0)
        label.append([0.0,1.0])

    for n in os.listdir(neg_file):
        img = cv2.imread(neg_file+'/'+n ,0)
        resize_img = cv2.resize(img, (70,134))

        data.append(resize_img/255.0)
        label.append([1.0,0.0])
    return np.array(data), np.array(label)

#------ 학습 데이터
data, label = read_file(pos_train_file, neg_train_file)
batch_images, batch_labels = tf.train.shuffle_batch([data, label],
                                                    batch_size=batch_size,
                                                    capacity=2000,
                                                 min_after_dequeue=1000,
                                                    enqueue_many=True)


#------- 검출을 위한 데이터
def get_data(num):
    data_file = './test'+str(num)+'.png'
    #원본이미지
    img = cv2.imread(data_file)

    #피라미드에서 자른 이미지와, 좌표데이터
    detect_data, detect_locate = pc.crop_image(data_file)
    return detect_data, detect_locate, img


with tf.Session() as sess:
    #-------- 배치큐를 위한 스레드 생성
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver = tf.train.Saver()
    initial_step = 0
    ckpt = tf.train.get_checkpoint_state(save_dir)
    sess.run(tf.global_variables_initializer())

    #---------- 훈련한 체크포인트 꺼내오기(에폭 500번, 95.3 정확도)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        # 복구한 시작 지점
        initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])
        print("Checkpoint")
        print(initial_step)
    else:
        print("No Checkpoint")

    #4개의 test 데이터를 모두 검사한다.
    for num in range(test_data_num):
        detect_data, detect_locate, img = get_data(num+1)

        for d,l in zip(detect_data,detect_locate):
            is_person = result.eval(feed_dict = {x_input:np.array([d]), keep_prob:1.0})
            if(is_person):
                x, y, i = l
                i = 1/i
                #축소되었던 이미지 범위를 다시 원래대로 확대해준다.
                cv2.rectangle(img, (int(y*i),int(x*i)),(int((y+70)*i),int((x+134)*i)),(0,0,255),3)

        #영역을 표시한 이미지 파일 저장
        cv2.imwrite('f_test'+str(num)+'_0.5_46810.png',img)
    coord.request_stop()
    coord.join(threads)
