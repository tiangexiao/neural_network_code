#如何把mnist数据集用tfrecord来进行重写

import tensorflow as tf
import numpy as np

mnist = tf.contrib.learn.datasets.load_dataset('mnist')
train_data = mnist.train.images
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)



print(train_labels.shape)

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
# print(train_data[0].dtype) #float32
def write_tfrecord(write_path, data, labels):
    writer = tf.python_io.TFRecordWriter(write_path)
    for i in range(data.shape[0]):
        image = data[i].tostring()
        label = labels[i]

        example = tf.train.Example(features=tf.train.Features(
            feature={
                'label':_int64_feature(label),
                'image':_bytes_feature(image)
            }
        ))
        writer.write(example.SerializeToString())
    writer.close()
write_tfrecord(r'../tfrecord/mnist.tfrecords', train_data, train_labels)




