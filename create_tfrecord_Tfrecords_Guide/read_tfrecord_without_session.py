#从建立的tfrecord里面取出数据

import  numpy as np
import tensorflow as tf
import skimage.io as io
import matplotlib.pyplot as plt
tfrecord_filename = '../tfrecord/cat.tfrecords'
read_iterator = tf.python_io.tf_record_iterator(path=tfrecord_filename)

images_ndarray = []
for string_record in read_iterator:
    example = tf.train.Example()

    example.ParseFromString(string_record)

    height = int(example.features.feature['height'].int64_list.value[0])
    width = int(example.features.feature['width'].int64_list.value[0])
    img_string = (example.features.feature['image_raw'].bytes_list.value[0])
    label = int(example.features.feature['label'].int64_list.value[0])
    print(label)
    img_1d = np.fromstring(img_string, dtype=np.uint8)
    reconstructed_img = img_1d.reshape((height, width, -1))
    #img_string为bytes类型，img_1d和reconstructed_img均为ndarray类型

    images_ndarray.append(reconstructed_img)

# #可以尝试画出来图像
# io.imshow(images_ndarray[0])
# plt.show()
