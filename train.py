import cv2

from t_init import *
import random

TRAIN_PATH = 'train/'
VAILD_PATH = 'train/'
img_map = {}


def get_namelist(path):
    list = []
    for e in os.listdir(path):
        list.append(e.replace('.png', ''))
    random.shuffle(list)
    return list


TRAIN_LIST = get_namelist(TRAIN_PATH)
VAILD_LIST = get_namelist(VAILD_PATH)


def label2vector(label):
    vec = zeros(LABEL_LEN)
    label = label.split('-')
    color = int(label[0])
    status = label[1]
    # 前32位作状态 后8位作颜色
    vec[STATUS_LEN + color] = 1
    for i in range(32):
        if status[i] == '1':
            vec[31 - i] = 1
    return vec


def get_data_and_label(name, path, type='.png'):
    img = cv2.imread(path + name + type)
    x = img.flatten() / 255
    y = label2vector(name)
    return x, y


def get_next_batch(batch_size, path, list):
    batch_x = zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH * 3])
    batch_y = zeros([batch_size, LABEL_LEN])
    for i in range(batch_size):
        if len(list) == 0:
            list = get_namelist(path)
        name = list[-1]
        list.pop()
        batch_x[i, :], batch_y[i, :] = get_data_and_label(name, path)
    return batch_x, batch_y


def train_CNN():
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
    # sigmoid交叉熵
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
    # 自适应矩估计优化
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, MODELS_PATH)
        for step in range(3000):
            batch_x, batch_y = get_next_batch(32, TRAIN_PATH, TRAIN_LIST)
            _ = sess.run([optimizer], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.9})
            if step % 99 == 0:
                batch_x, batch_y = get_next_batch(128, VAILD_PATH, VAILD_LIST)
                _acc = sess.run([accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1})
                print(_acc)

        saver.save(sess, MODELS_PATH)
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                        output_node_names=['predict'])
        with tf.gfile.GFile("D:/Personal/Desktop/project/Android/backup/app/src/main/assets/models.pb", mode='wb') as f:
            f.write(output_graph_def.SerializeToString())


train_CNN()
