#建立的tfrecord里面迭代取出数据，并使用shuffle和batch
#http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

import  numpy as np
import tensorflow as tf
import skimage.io as io
import matplotlib.pyplot as plt
tfrecords_filename  = '../tfrecord/cat.tfrecords'
IMAGE_HEIGHT = IMAGE_WIDTH = 384

def read_and_decode(filename_queue):
    filename_queue = tf.train.string_input_producer([filename_queue], num_epochs=10)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    label = tf.cast(features['label'], tf.int32)

    image = tf.reshape(image, [height, width, 3])

    # image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.int32)

    resized_image = tf.image.resize_image_with_crop_or_pad(image=image, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)
    images, label = tf.train.shuffle_batch([resized_image, label], batch_size=2, capacity=30, num_threads=2, min_after_dequeue=10)

    return images, label


image, label = read_and_decode(tfrecords_filename)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)


    for i in range(3):
        img, lab = sess.run([image, label])
        print('current batch')
        io.imshow(img[0,:,:,:])
        io.show()
        io.imshow(img[1,:,:,:])
        io.show()


    coord.request_stop()
    coord.join(threads)
