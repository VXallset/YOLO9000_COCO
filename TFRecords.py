"""
This file is used to generate the TFrecords of the COCO dataset

By: Yu Sun
vxallset@outlook.com
Last modified: Dec 1st, 2018
All right reserved.
"""
from random import shuffle
import glob
import os
import cv2
from skimage import io
import numpy as np
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def read_image(address, data_type=np.uint8):
    """
    This function is used for reading an image from the given address.
    :param address: String. The address of the image to be read.
    :param data_type: default: np.uint8.  The data type of image.
    :return: img_id: String img: ndarray, img_shape:Tuple. The image and its id and shape
    """
    f = tf.gfile.FastGFile(address, 'rb')

    img = cv2.imread(address)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(data_type)
    #img_name = address.split('/')[-1]
    img_id, _ = os.path.splitext(address.split('/')[-1])
    img_id = int(img_id)
    img_shape = img.shape
    if len(img.shape) == 2:
        img_shape[2] = 1
    img = f.read()
    return img_id, img, img_shape


def encode_to_tfrecords(input_dir='./', output_dir='./', tfrecords_name='data.tfrecords', shuffle_data=True,
                        split_data=False, train_percentage=0.6, validation_percentage=0.2, test_percentage=0.2, data_type=np.uint8):
    """
    This function can make a TFRecord using the images in the input file path and save it to the output file path.

    :param input_dir: String, default = './'. The file path which contain the images of the dataset.
    :param output_dir: String, default = './'. The file path which will be used to save the TFRecord.
    :param tfrecords_name: String, default = 'data.tfrecords'.
    :param shuffle_data: Bool, default = True. Whether the data will be shuffled. If =True, the address will be shuffled
    before saving.
    :param split_data: Bool, default = False, Whether the data will be split into the training, validating, and testing sets.
    :param train_percentage: Float, default = 0.6. The percentage of images which will be used as train data.
    :param validation_percentage: Float, default = 0.2. The percentage of images which will be used as validation data.
    :param test_percentage: Float, default = 0.2. The percentage of images which will be used as test data.
    :param data_type: the type of image, default= np.uint8.
    :return: Int. 0: Success. -1: The values of train_percentage & validation_percentage & test_percentage error. 1:
    Input error.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('output_dir is not exist, make new directory:', output_dir)

    if not os.path.exists(input_dir):
        print('input_dir not exists!')
        return 1

    if train_percentage + validation_percentage + test_percentage != 1.0:
        print('train_percentage + validation_percentage + test_percentage != 1.0. Please use the default setting or ',
              'correct the values of them.')
        return -1

    # list all the image files in the folder.
    addrs = glob.glob(os.path.join(input_dir, '*'))

    # shuffle the address of the image files.
    if shuffle_data:
        shuffle(addrs)

    if split_data:
        addrs_percentage = [0, train_percentage, train_percentage + validation_percentage, 1]

        filenames = ['train.tfrecords', 'validation.tfrecords', 'test.tfrecords']

        for filei in range(len(filenames)):
            print('====================Convert file: {}/{}======================'.format(filei + 1, len(filenames)))
            filename = os.path.join(output_dir, filenames[filei])
            data_addrs = addrs[int(addrs_percentage[filei] * len(addrs)): int(addrs_percentage[filei + 1] * len(addrs))]

            with tf.python_io.TFRecordWriter(filename) as writer:
                for i in range(len(data_addrs)):
                    if i % 1000 == 1:
                        print('Convert data: {}/{}'.format(i, len(data_addrs)))
                        # read the image
                    img_id, img, img_shape = read_image(data_addrs[i], data_type=data_type)

                    img_shape0 = img_shape[0]
                    img_shape1 = img_shape[1]
                    img_shape2 = 1
                    # some images may only have 2 channels, in this case, we make the 3rd dimension to be 1
                    if len(img_shape) >= 3:
                        img_shape2 = img_shape[2]

                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'image_id': _int64_feature(img_id),
                                'image_raw': _bytes_feature(img),
                                'height': _int64_feature(img_shape0),
                                'width': _int64_feature(img_shape1),
                                'channels': _int64_feature(img_shape2)
                            }))
                    writer.write(example.SerializeToString())
        print('Done!')
    else:
        filename = os.path.join(output_dir, tfrecords_name)  # address to save the TFRecords file
        # open the TFRecords file
        with tf.python_io.TFRecordWriter(filename) as writer:
            for i in range(len(addrs)):
                # print how many images are saved every 1000 images
                if i % 1000 == 1:
                    print('Convert data: {}/{}'.format(i, len(addrs)))

                # read the image
                img_id, img, img_shape = read_image(addrs[i], data_type=data_type)

                img_shape0 = img_shape[0]
                img_shape1 = img_shape[1]
                img_shape2 = 1
                # some images may only have 2 channels, in this case, we make the 3rd dimension to be 1
                if len(img_shape) >= 3:
                    img_shape2 = img_shape[2]

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image_id': _int64_feature(img_id),
                            'image_raw': _bytes_feature(img),
                            'height': _int64_feature(img_shape0),
                            'width': _int64_feature(img_shape1),
                            'channels': _int64_feature(img_shape2)
                        }))
                writer.write(example.SerializeToString())
        print('Done!')
    return 0


def decode_from_tfrecords(filename_queue, data_type=np.uint8):
    """
    This function is used to decode a TFRecords file.

    :param filename_queue: FIFOQueue object
    :param data_type: default=np.uint8. The data type of images saved in the .tfrecord file.
    :return: image, image_id, image_shape. image: an image. image_id: the file name of the image, image_shape: the shape
    of the image
    """

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={'image_id': tf.FixedLenFeature([], tf.int64),
                                                 'image_raw': tf.FixedLenFeature([], tf.string),
                                                 'height': tf.FixedLenFeature([], tf.int64),
                                                 'width': tf.FixedLenFeature([], tf.int64),
                                                 'channels': tf.FixedLenFeature([], tf.int64)})

    image = tf.image.decode_image(features['image_raw'], dtype=data_type)

    # image = tf.decode_raw(features['image_raw'], data_type)
    image = tf.reshape(image, [features['height'], features['width'], features['channels']])
    #image_id = tf.cast(features['image_id'], tf.string)
    image_id = features['image_id']
    image_shape = (features['height'], features['width'], features['channels'])

    return image, image_id, image_shape


def read_image_v2(address, data_type=np.uint8):
    """
    This function is used for reading an image from the given address. However, it works a little bit different from the
    read_image() function. This function return bytes image, therefore, we should use different way to write and read
    the TFRecords file.

    # using the following code to decode images saved in the TFRecords file if we use the read_image_v2() function to
    # read images from the dataset.
    :param address: String. The address of the image to be read.
    :param data_type: default: np.uint8.  The data type of image.
    :return: img_id: String img: bytes, img_shape:Tuple. The image and its id and shape
    """
    img = io.imread(address)
    img = img.astype(data_type)
    img_id, _ = os.path.splitext(address.split('/')[-1])
    img_id = int(img_id)
    img_shape = img.shape
    return img_id, img, img_shape


def encode_to_tfrecords_v2(input_dir='./', output_dir='./', tfrecords_name='data.tfrecords', shuffle_data=True, split_data=False, train_percentage=0.6,
                           validation_percentage=0.2, test_percentage=0.2, data_type=np.uint8):
    """
    This function can make a TFRecord using the images in the input file path and save it to the output file path.

    :param input_dir: String, default = './'. The file path which contain the images of the dataset.
    :param output_dir: String, default = './'. The file path which will be used to save the TFRecord.
    :param tfrecords_name: String, default = 'data.tfrecords'.
    :param shuffle_data: Bool, default = True. Whether the data will be shuffled. If =True, the address will be shuffled
    before saving.
    :param split_data: Bool, default = False, Whether the data will be split into the training, validating, and testing sets.
    :param train_percentage: Float, default = 0.6. The percentage of images which will be used as train data.
    :param validation_percentage: Float, default = 0.2. The percentage of images which will be used as validation data.
    :param test_percentage: Float, default = 0.2. The percentage of images which will be used as test data.
    :param data_type: the type of image, default= np.uint8.
    :return: Int. 0: Success. -1: The values of train_percentage & validation_percentage & test_percentage error. 1:
    Input error.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('output_dir is not exist, make new directory:', output_dir)

    if not os.path.exists(input_dir):
        print('input_dir not exists!')
        return 1

    if train_percentage + validation_percentage + test_percentage != 1.0:
        print('train_percentage + validation_percentage + test_percentage != 1.0. Please use the default setting or ',
              'correct the values of them.')
        return -1

    # list all the image files in the folder.
    addrs = glob.glob(os.path.join(input_dir, '*'))

    # shuffle the address of the image files.
    if shuffle_data:
        shuffle(addrs)

    if split_data:
        addrs_percentage = [0, train_percentage, train_percentage + validation_percentage, 1]

        filenames = ['train.tfrecords', 'validation.tfrecords', 'test.tfrecords']

        for filei in range(len(filenames)):
            print('====================Convert file: {}/{}======================'.format(filei + 1, len(filenames)))
            filename = os.path.join(output_dir, filenames[filei])
            data_addrs = addrs[int(addrs_percentage[filei] * len(addrs)): int(addrs_percentage[filei + 1] * len(addrs))]

            with tf.python_io.TFRecordWriter(filename) as writer:
                for i in range(len(data_addrs)):
                    if i % 1000 == 1:
                        print('Convert data: {}/{}'.format(i, len(data_addrs)))
                        # read the image
                    img_id, img, img_shape = read_image_v2(data_addrs[i], data_type=data_type)

                    img_shape0 = img_shape[0]
                    img_shape1 = img_shape[1]
                    img_shape2 = 1
                    # some images may only have 2 channels, in this case, we make the 3rd dimension to be 1
                    if len(img_shape) >= 3:
                        img_shape2 = img_shape[2]

                    img = img.tostring()
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'image_id': _int64_feature(img_id),
                                'image_raw': _bytes_feature(img),
                                'height': _int64_feature(img_shape0),
                                'width': _int64_feature(img_shape1),
                                'channels': _int64_feature(img_shape2)
                            }))
                    writer.write(example.SerializeToString())
        print('Done!')
    else:
        filename = os.path.join(output_dir, tfrecords_name)  # address to save the TFRecords file
        # open the TFRecords file
        with tf.python_io.TFRecordWriter(filename) as writer:
            for i in range(len(addrs)):
                # print how many images are saved every 1000 images
                if i % 1000 == 1:
                    print('Convert data: {}/{}'.format(i, len(addrs)))

                # read the image
                img_id, img, img_shape = read_image_v2(addrs[i], data_type=data_type)

                img_shape0 = img_shape[0]
                img_shape1 = img_shape[1]
                img_shape2 = 1
                # some images may only have 2 channels, in this case, we make the 3rd dimension to be 1
                if len(img_shape) >= 3:
                    img_shape2 = img_shape[2]

                img = img.tostring()
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image_id': _int64_feature(img_id),
                            'image_raw': _bytes_feature(img),
                            'height': _int64_feature(img_shape0),
                            'width': _int64_feature(img_shape1),
                            'channels': _int64_feature(img_shape2)
                        }))
                writer.write(example.SerializeToString())
        print('Done!')
    return 0


def decode_from_tfrecords_v2(filename_queue, data_type=np.uint8):
    """
    This function is used to decode a TFRecords file.

    :param filename_queue: FIFOQueue object
    :param data_type: default=np.uint8. The data type of images saved in the .tfrecord file.
    :return: image, image_id, image_shape. image: an image. image_id: the file name of the image, image_shape: the shape
    of the image
    """

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={'image_id': tf.FixedLenFeature([], tf.int64),
                                                 'image_raw': tf.FixedLenFeature([], tf.string),
                                                 'height': tf.FixedLenFeature([], tf.int64),
                                                 'width': tf.FixedLenFeature([], tf.int64),
                                                 'channels': tf.FixedLenFeature([], tf.int64)})

    image = tf.decode_raw(features['image_raw'], data_type)
    image = tf.reshape(image, [features['height'], features['width'], features['channels']])
    #image_id = tf.cast(features['image_id'], tf.string)
    image_id = features['image_id']
    image_shape = (features['height'], features['width'], features['channels'])

    return image, image_id, image_shape


def test( is_creat_TFRecords=True, is_split_data=False, TFfilename='data.tfrecords'):
    """
    This function is used to test whether the other functions can work properly
    :return:
    """
    if is_creat_TFRecords:
        encode_to_tfrecords('../dataset/COCO/val2017/', '../dataset/COCO/TFRecords/', split_data=is_split_data)
        if is_split_data:
            TFfilename='train.tfrecords'
        else:
            TFfilename='data.tfrecords'
    else:
        if not os.path.exists('../dataset/COCO/TFRecords/train.tfrecords') and not os.path.exists(
                '../dataset/COCO/TFRecords/data.tfrecords'):
            print('No TFRecords! Creating now...')
            encode_to_tfrecords('../dataset/COCO/val2017/', '../dataset/COCO/TFRecords/', split_data=False)
            TFfilename='data.tfrecords'

    filename_queue = tf.train.string_input_producer(['../dataset/COCO/TFRecords/'+TFfilename], num_epochs=None)
    image, image_id, image_shape = decode_from_tfrecords(filename_queue)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            # while not coord.should_stop():
            for i in range(1):
                img, imgid, imgshape = sess.run([image, image_id, image_shape])
                io.imshow(img)
                io.show()
                print(type(img[0,0,0]), imgid, imgshape)
                print('img id type:',str(type(imgid)))

        except tf.errors.OutOfRangeError:
            print('Done reading')
        finally:
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)

def test_v2(is_creat_TFRecords=True, is_split_data=True, TFfilename='data.tfrecords'):
    """
    This function is used to test whether the other functions can work properly
    :return:
    """
    if is_creat_TFRecords:
        encode_to_tfrecords('../dataset/COCO/val2017/', '../dataset/COCO/TFRecords/', split_data=is_split_data)
        if is_split_data:
            TFfilename='train.tfrecords'
        else:
            TFfilename='data.tfrecords'
    else:
        if not os.path.exists('../dataset/COCO/TFRecords/train.tfrecords') and not os.path.exists(
                '../dataset/COCO/TFRecords/data.tfrecords'):
            print('No TFRecords! Creating now...')
            encode_to_tfrecords('../dataset/COCO/val2017/', '../dataset/COCO/TFRecords/', split_data=False)
            TFfilename='data.tfrecords'
    if is_creat_TFRecords:
        encode_to_tfrecords_v2('../dataset/COCO/val2017/', '../dataset/COCO/TFRecords/', split_data=is_split_data)

    filename_queue = tf.train.string_input_producer(['../dataset/COCO/TFRecords/'+TFfilename], num_epochs=None)
    image, image_id, image_shape = decode_from_tfrecords_v2(filename_queue)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            # while not coord.should_stop():
            for i in range(1):
                img, imgid, imgshape = sess.run([image, image_id, image_shape])
                io.imshow(img)
                io.show()
                print(img, imgid, imgshape)

        except tf.errors.OutOfRangeError:
            print('Done reading')
        finally:
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)

#test(is_creat_TFRecords=True, is_split_data=False, TFfilename='val.tfrecords')

encode_to_tfrecords('../dataset/COCO/val2017/', output_dir='../dataset/COCO/TFRecords/', tfrecords_name='val2017.tfrecords')

#test(False, False,TFfilename='val2017.tfrecords')