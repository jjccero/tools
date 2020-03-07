from t_init import *

import time
import cv2

saver = tf.train.Saver()


def crack_captcha(batch_x):
    with tf.Session() as sess:
        saver.restore(sess, MODELS_PATH)
        _color, _status = sess.run([pcolor, pstatus], feed_dict={X: batch_x, keep_prob: 1})
        sess.close()
    return _color, _status


imgfilename = 'temp.png'

SUB_WIDTH = 36
SRC_WIDTH = SUB_WIDTH * 6
SRC_HEIGHT = 384

sy = (SRC_HEIGHT - SUB_WIDTH) // 2


def vector2label(_color, _status):
    rcolor = [0] * 30
    rstatus = [0] * 30
    for e1, e2 in _color:
        rcolor[e1] = e2

    for e1, e2 in _status:
        rstatus[e1] |= 1 << (31 - e2)

    with open('stone', 'w') as f:
        for i in range(30):
            line = str(rcolor[i]) + ' ' + str(rstatus[i])
            f.writelines(line+'\n')
        f.close()
    print('识别成功')

def readfromjava():
    img = cv2.imread(imgfilename)
    img = cv2.resize(img, (SRC_WIDTH, SRC_HEIGHT))
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    batch_x = zeros([30, IMAGE_WIDTH * IMAGE_HEIGHT*3])
    for i in range(5):
        for j in range(6):
            subimg = img[sy + i * SUB_WIDTH:sy + (i + 1) * SUB_WIDTH, j * SUB_WIDTH:(j + 1) * SUB_WIDTH,:]
            batch_x[6 * i + j:] = subimg.flatten() / 255
    _color, _status = crack_captcha(batch_x)
    vector2label(_color, _status)

def main():
    waittime = int(input('扫描时间(s):'))
    while True:
        try:
            if os.path.exists(imgfilename):
                with open(imgfilename) as f:
                    readfromjava()
                os.remove(imgfilename)

        except:
            print('有点卡')

        time.sleep(waittime)


print('CNN初始化完成')
main()
