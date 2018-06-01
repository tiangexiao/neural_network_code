#如何使用skimage.io 将一个图片读成 numpy.ndarray
#将ndarray转换为bytes类型
#从一个bytes类型重建为一个ndarray类型
#url: http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

import numpy as np

import skimage.io as io
import tensorflow as tf
from PIL import Image

cat1_path = r'../input/cats/cat1.jpg'

cat_img = io.imread(cat1_path)  #type:<class 'numpy.ndarray'> shape:(183, 276, 3)

io.imshow(cat_img)
io.show()  #显示图片

cat_string = cat_img.tostring()  #type:<class 'bytes'>

reconstructed_cat_1d = np.fromstring(cat_string, dtype=np.uint8) #type:<class 'numpy.ndarray'> shape:(151524,)
reconstructed_cat_img = reconstructed_cat_1d.reshape(cat_img.shape)  #type: <class 'numpy.ndarray'> shape:(183, 276, 3)

np.allclose(cat_img, reconstructed_cat_img) #判断两个类型是否相等  result：ture


