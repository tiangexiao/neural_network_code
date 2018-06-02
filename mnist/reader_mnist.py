#如何把mnist数据集用tfrecord来进行重写 

import numpy as np
import tensorflow as tf
import sklearn
tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here



def cnn_model_fn(features, labels, mode):
    # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    input_layer = tf.reshape(features, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    dense=tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu
    )

    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        'classes' : tf.argmax(input=logits, axis=1),
        'probabilities' : tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op=train_op)

    eval_metric_ops={
        'accuracy':tf.metrics.accuracy(
            labels=labels,
            predictions=predictions['classes']
        )
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)






#create the estimator
mnist_estimator = tf.estimator.Estimator(
    model_fn = cnn_model_fn, model_dir='../output'
)


#set up logging for prediction
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50
)

#train the model



def my_read_and_decode(filename, random_crop=False, random_clip=False, shuffle_batch=True):

    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image' : tf.FixedLenFeature([], tf.string),
                                       })

    image = tf.decode_raw(features['image'], tf.float32)  #这里是float32,不是uint8,因为存的时候为float32
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [28*28])

    label = tf.cast(features['label'], tf.int32)

    label = tf.reshape(label, [1])

    if shuffle_batch:
        image, label = tf.train.shuffle_batch([image, label],batch_size=100,capacity=8000,num_threads=4,min_after_dequeue=2000)
    else:
        image, label = tf.train.batch([image,  label],batch_size=1, capacity=8000,num_threads=4)

    return image, label

train_input_fn = lambda :my_read_and_decode(r'../tfrecord/mnist.tfrecords')
mnist_estimator.train(input_fn=train_input_fn,
                      steps=20000,
                      hooks=[logging_hook])


