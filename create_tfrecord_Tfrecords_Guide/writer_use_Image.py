#使用Image读取图片，并且保存
#url: https://blog.csdn.net/qq_39037910/article/details/72900308
from PIL import Image
import tensorflow as tf


cat1_path = r'../input/cats/cat1.jpg'
cat2_path = r'../input/cats/cat2.jpg'
cat3_path = r'../input/cats/cat3.jpg'
cat4_path = r'../input/cats/cat4.jpg'
cat_paths = [cat1_path, cat2_path, cat3_path, cat4_path]

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

writer = tf.python_io.TFRecordWriter('../tfrecord/cats.tfrecords')
for cat_path in cat_paths:
    image = Image.open(cat_path) #这里和用numpy方法都用了Image，但是还是有些不同的，numpy是把数据当做数组来进行处理，而这里是直接使用image对象
    rows = image.size[0]
    cols = image.size[1]
    image_raw = image.tobytes()   #实际上这种方法和

    example = tf.train.Example(features=tf.train.Features(
        feature={
            'height': _int64_feature(rows),
            'weight': _int64_feature(cols),
            'label': _int64_feature(0),
            'image_raw': _bytes_feature(image_raw)
        }
    ))
    writer.write(example.SerializeToString())
writer.close()
