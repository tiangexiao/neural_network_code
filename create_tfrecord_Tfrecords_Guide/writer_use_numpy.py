#如何写入tfrecord
#使用了numpy来保存图片的数据
#url: http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/


import numpy as np

import skimage.io as io
import tensorflow as tf
from PIL import Image

cat1_path = r'../input/cats/cat1.jpg'
cat2_path = r'../input/cats/cat2.jpg'
cat3_path = r'../input/cats/cat3.jpg'
cat4_path = r'../input/cats/cat4.jpg'
cat_paths = [cat1_path, cat2_path, cat3_path, cat4_path]


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
tfrecords_filename = '../tfrecord/cat.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)


for cat_path in cat_paths:
    img = np.array(Image.open(cat_path))  #实际上这条语句等于 img = io.imread(cat_path)

    height = img.shape[0]
    weight = img.shape[1]

    img_raw = img.tostring()

    example = tf.train.Example(features=tf.train.Features(
        feature={
            'height': _int64_feature(height),
            'width': _int64_feature(weight),
            'image_raw': _bytes_feature(img_raw),
            'label': _int64_feature(0)
        }
    ))

    writer.write(example.SerializeToString())
writer.close()
