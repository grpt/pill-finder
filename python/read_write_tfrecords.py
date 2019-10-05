import tensorflow as tf
import numpy as np
import cv2
from make_square import make_square

def write_tfrecords(directory,tfrecords_name,label):
    image_list = []
    label_list = []

    size = 48
    n_pic = 40000
    for i in range(1,n_pic+1):
        name = directory +'_' +str(i) + '.jpg'
        image_list.append(name)
        label_list.append(label)

    options = tf.python_io.TFRecordOptions(compression_type = tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter('./tfrecords/'+tfrecords_name)

    for image_name,label in zip(image_list,label_list):
        try:
            image = cv2.imread(image_name,cv2.IMREAD_GRAYSCALE)
            image = make_square(image)
            image = cv2.resize(image,(size,size))

            _image = image.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'image' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [_image])),
                'label' : tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
            }))

            writer.write(example.SerializeToString())
            print(image_name,"finished")
        except:
            continue

    writer.close()

def _parse_function(example_proto):
    size = 48

    features = {'image': tf.FixedLenFeature([], tf.string, default_value=""),
              'label': tf.FixedLenFeature([], tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features=features)
    image = tf.decode_raw(parsed_features['image'],tf.uint8)
    image = tf.reshape(image,[size,size,1])
    label = parsed_features['label']
    return image, label


def read_tfrecords(tfrecords_name):
    filenames = [tfrecords_name]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)

    return dataset


# shape_name = ['one', 'ellipse', 'etc']
shape_name = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','default']

# for label in range(len(shape_name)):
#     directory = './char/' + shape_name[label] +'/'
#     write_tfrecords(directory, shape_name[label]+'.tfrecords',label)