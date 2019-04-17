# -*- coding: utf-8 -*-
import tensorflow as tf

#创建一张2通道图
img = tf.constant(value=[[[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]]]],dtype=tf.float32)
img = tf.concat(values=[img,img],axis=3)

#3*3卷积核
filter = tf.constant(value=1, shape=[3,3,2,5], dtype=tf.float32)
#DCNN
#out_img = tf.nn.atrous_conv2d(value=img, filters=filter, rate=1)

out_img1 = tf.nn.atrous_conv2d(value=img, filters=filter, rate=1, padding='SAME')
out_img2 = tf.nn.atrous_conv2d(value=img, filters=filter, rate=1, padding='VALID')
out_img3 = tf.nn.atrous_conv2d(value=img, filters=filter, rate=2, padding='SAME')

#error
#out_img4 = tf.nn.atrous_conv2d(value=img, filters=filter, rate=2, padding='VALID')

with tf.Session() as sess:
    print('rate=1, SAME mode result:')
    print(sess.run(out_img1))

    print('rate=1, VALID mode result:')
    print(sess.run(out_img2))

    print('rate=2, SAME mode result:')
    print(sess.run(out_img3))

    # error
    #print 'rate=2, VALID mode result:'
    #print(sess.run(out_img4))





