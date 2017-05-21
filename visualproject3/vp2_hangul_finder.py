import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt

#19600개
with h5py.File('kalph_train_add__.hf', 'r') as hf:
    images = np.array(hf['images'])
    labels = np.array(hf['labels'])


with h5py.File('kalph_test.hf', 'r') as hf:
    test_images = np.array(hf['images'])
    test_labels = np.array(hf['labels'])


data_size = images.shape[0]
print(data_size)
batch_size = 20
image_size = 52
input_size = image_size * image_size
target_size = 14
filter_size = 5
dep1 = 32
dep2 = 64
dep3 = 128
fc_dep1 = 1024


x_input = tf.placeholder(tf.float32, [None, input_size])
y_input = tf.placeholder(tf.float32, [None, target_size])

#아 주 중 요 하 다
#weight의 초기화에 따라서
#(0으로 했을 경우에) dead neurons 가 발생할 수 있다.

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

#demension이 결정되지 않았다.
x_image = tf.reshape(x_input , [-1, image_size, image_size, 1])

#채널 : 1 ,출력
W_conv1 = tf.get_variable(name="weight_conv1",initializer=initializer, shape=[filter_size, filter_size, 1, dep1])
b_conv1 = bias_variable([dep1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


#2층
W_conv2 = tf.get_variable(name="weight_conv2",initializer=initializer, shape=[filter_size, filter_size, dep1, dep2])
b_conv2 = bias_variable([dep2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#3층
W_conv3 = tf.get_variable(name="weight_conv3",initializer=initializer, shape=[filter_size, filter_size, dep2, dep3])
b_conv3 = bias_variable([dep3])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

fc1 = 7*7*dep3

#fully connected
W_fc1 = tf.get_variable(name="weight_fully_connected1",initializer=initializer, shape=[fc1, fc_dep1])
b_fc1 = bias_variable([fc_dep1])

h_pool3_flat = tf.reshape(h_pool3, [-1, fc1])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = tf.get_variable(name="weight_fully_connected2",initializer=initializer, shape=[fc_dep1, target_size])
b_fc2 = bias_variable([target_size])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2



cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_input, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



try:
    batch_images, batch_labels = tf.train.batch([images, labels],
                                                batch_size=batch_size,
                                                enqueue_many=True,
                                                capacity=3)
    batch_images = tf.reshape(batch_images, [batch_size, -1])
    batch_labels = tf.one_hot(batch_labels, depth = 14, on_value=1.0, off_value=0.0,axis=-1)

    t_batch_images, t_batch_labels = tf.train.batch([test_images, test_labels],
                                                batch_size=batch_size,
                                                enqueue_many=True,
                                                capacity=3)
    t_batch_images = tf.reshape(t_batch_images, [batch_size, -1])
    t_batch_labels = tf.one_hot(t_batch_labels, depth = 14, on_value=1.0, off_value=0.0,axis=-1)
    save_dir = './tmp2'
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.trainable_variables())
        initial_step = 0
        ckpt = tf.train.get_checkpoint_state(save_dir)

        #checkpoint가 존재할 경우 변수 값을 복구한다.
        """
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            #복구한 시작 지점
            initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])
            print("Checkpoint")
            print(initial_step)
        else:
            print("No Checkpoint")

        """
        for epoch in range(initial_step, 27):
            print(epoch,'-------------')
            for i in range(int(data_size/batch_size)):
                b_i, b_l = sess.run([batch_images, batch_labels])
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x_input: b_i, y_input:b_l, keep_prob: 1.0})
                    print('step', i, 'training accuracy', train_accuracy)
                train_step.run(feed_dict = {x_input:b_i, y_input:b_l, keep_prob :0.5})

            saver.save(sess, save_dir+'.ckpt', global_step = epoch)

            count = 0
            all_accuracy = 0

            for i in range(100):
                count += 1

                t_i, t_l = sess.run([t_batch_images, t_batch_labels])
                train_accuracy = accuracy.eval(feed_dict={x_input: t_i, y_input: t_l, keep_prob: 1.0})
                all_accuracy += train_accuracy
            print(all_accuracy/count)
except:
    print("ㅎㅎ망할^^")