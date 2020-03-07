import tensorflow as tf
from numpy import zeros, array
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
IMAGE_WIDTH = 36
IMAGE_HEIGHT = 36

COLOR_LEN = 8
STATUS_LEN = 32
LABEL_LEN = COLOR_LEN + STATUS_LEN

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH*3],name='input_x')
Y = tf.placeholder(tf.float32, [None, LABEL_LEN])
keep_prob = tf.placeholder(tf.float32,name='keep_prob')  # dropout


# 定义CNN
def stone_CNN(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 3, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 32]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # 全连接层
    w_d = tf.Variable(w_alpha * tf.random_normal([IMAGE_WIDTH // 4 * IMAGE_HEIGHT // 4 * 32, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv2, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, LABEL_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([LABEL_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


output = stone_CNN()
predict = tf.reshape(tf.sigmoid(tf.reshape(output, [-1, LABEL_LEN])) > 0.9,[-1, LABEL_LEN],name='predict')

ly = tf.reshape(Y>0.9, [-1, LABEL_LEN])
correct_pred =tf.reduce_all(tf.equal(predict, ly),1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

MODELS_PATH = 'models/jjcero_Auto.model'

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"`
