"""
This is used for object detection, implemented using Yolo9000 with some of modification (mainly using different loss
function). The model is trained using COCO dataset and the darknet-19.

By: Yu Sun
vxallset@outlook.com
Last modified: Dec 1st, 2018
All rights reserved.

"""
import os
from pycocotools.coco import COCO
import tensorflow as tf
import numpy as np
import dataset
import lossfunction
import time
import cv2


# ================================================================================================================
# activation function
def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.1, name='leaky_relu') # or using tf.maximum(0.1*x,x)


# Conv+BN：all convolution layers in yolo 9000 netwoork are connected with a BN layer
def conv2d(x, filters_num, filters_size, pad_size=0, stride=1, batch_normalize=True,
           activation=leaky_relu, use_bias=False, name='conv2d'):
    # please do not use padding="SAME", otherwise the coordinate may be incorrect
    if pad_size > 0:
        x = tf.pad(x, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    out = tf.layers.conv2d(x, filters=filters_num, kernel_size=filters_size, strides=stride,
                           padding='VALID', activation=None, use_bias=use_bias, name=name)
    if batch_normalize:
        out = tf.layers.batch_normalization(out, axis=-1, momentum=0.9, training=False, name=name+'_bn')
    if activation:
        out = activation(out)
    return out


# max_pool
def maxpool(x, size=2, stride=2, name='maxpool'):
    return tf.layers.max_pooling2d(x, pool_size=size, strides=stride)


# reorg layer
def reorg(x, stride):
    return tf.space_to_depth(x, block_size=stride)
    # or using return tf.extract_image_patches(x,ksizes=[1,stride,stride,1],strides=[1,stride,stride,1],
    # rates=[1,1,1,1],padding='VALID')


# --------------------------------------------  Darknet19  --------------------------------------------------------
# the number of channels in the last layer is  anchor_num * (class_num +5 ) = 5 * (80 + 5) = 425


def darknet(images,n_last_channels=425):
    net = conv2d(images, filters_num=32, filters_size=3, pad_size=1, name='conv1')
    net = maxpool(net, size=2, stride=2, name='pool1')

    net = conv2d(net, 64, 3, 1, name='conv2')
    net = maxpool(net, 2, 2, name='pool2')

    net = conv2d(net, 128, 3, 1, name='conv3_1')
    net = conv2d(net, 64, 1, 0, name='conv3_2')
    net = conv2d(net, 128, 3, 1, name='conv3_3')
    net = maxpool(net, 2, 2, name='pool3')

    net = conv2d(net, 256, 3, 1, name='conv4_1')
    net = conv2d(net, 128, 1, 0, name='conv4_2')
    net = conv2d(net, 256, 3, 1, name='conv4_3')
    net = maxpool(net, 2, 2, name='pool4')

    net = conv2d(net, 512, 3, 1, name='conv5_1')
    net = conv2d(net, 256, 1, 0,name='conv5_2')
    net = conv2d(net,512, 3, 1, name='conv5_3')
    net = conv2d(net, 256, 1, 0, name='conv5_4')
    net = conv2d(net, 512, 3, 1, name='conv5_5')
    # save this layer to construct passthrough layer later
    shortcut = net
    net = maxpool(net, 2, 2, name='pool5')

    net = conv2d(net, 1024, 3, 1, name='conv6_1')
    net = conv2d(net, 512, 1, 0, name='conv6_2')
    net = conv2d(net, 1024, 3, 1, name='conv6_3')
    net = conv2d(net, 512, 1, 0, name='conv6_4')
    net = conv2d(net, 1024, 3, 1, name='conv6_5')

    net = conv2d(net, 1024, 3, 1, name='conv7_1')
    net = conv2d(net, 1024, 3, 1, name='conv7_2')
    # add a conv layer and the go passthrough
    # 26 x 26 x 512 -> 26 x 26 x 64 -> 13 x 13 x 256 feature map
    shortcut = conv2d(shortcut, 64, 1, 0, name='conv_shortcut')
    shortcut = reorg(shortcut, 2)
    # concat the channel with the shortcut
    net = tf.concat([shortcut, net], axis=-1)
    net = conv2d(net, 1024, 3, 1, name='conv8')

    # detection layer: no BN or activation function
    output = conv2d(net, filters_num=n_last_channels, filters_size=1, batch_normalize=False,
                 activation=None, use_bias=True, name='conv_dec')

    return output

def _main():
    """
    main function
    :return:
    """
    show_all_parameters = True

    batch_size = 16
    num_epochs = 20
    # save the model per save_epoch_number epochs
    save_epoch_number = 1
    display_step = 100
    istrain = True


    model_folder = './model/'
    img_folder = './images/'

    if istrain:
        datasetname = '../dataset/COCO/TFRecords/train2017.tfrecords'
        annotationFile = '../dataset/COCO/annotations_trainval2017/annotations/instances_train2017.json'

        #datasetname = '/home/cv/YuSun/Yolo9000/TFRecords/train2017.tfrecords'
        #annotationFile = '/home/cv/dataset/coco/annotations/instances_train2017.json'
        modelfile = './model/model.ckpt'
        image_numbers = 118287
    else:
        datasetname = '../dataset/COCO/TFRecords/val2017.tfrecords'
        annotationFile = '../dataset/COCO/annotations_trainval2017/annotations/instances_val2017.json'

        #datasetname = '/home/cv/YuSun/Yolo9000/TFRecords/val2017.tfrecords'
        #annotationFile = '/home/cv/dataset/coco/annotations/instances_val2017.json'
        modelfile = './model/epoch10.ckpt-81312'
        image_numbers = 5000

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    if not os.path.exists(datasetname):
        print('Dataset file (TFRecords) {} not exist!!!'.format(datasetname))
    if not os.path.exists(annotationFile):
        print('Annotation file {} not exist!!!'.format(annotationFile))
    if not os.path.exists(modelfile + '.index'):
        if not istrain:
            print('Model file {} not exist!!!'.format(modelfile))


    if show_all_parameters:
        print('------------------------------------------------------------------------------------------------')
        print('dataset name:{}'.format(datasetname))
        print('annotation file name:{}'.format(annotationFile))
        print('batch size = {}'.format(batch_size))
        print('istrain = {}'.format(istrain))
        if istrain:
            print('epoch num = {}'.format(num_epochs))
            print('Model will be saved in {} for every {} epochs'.format(model_folder, save_epoch_number))
            print('The result of loss will be shown for every {} steps.'.format(display_step))
        else:
            print('The model used to test is {}'.format(modelfile))
        print('------------------------------------------------------------------------------------------------')


    #------------------------------------------------------------------------------------------------------------------


    images_batch, ids_batch, shapes_batch = dataset.input_batch(datasetname=datasetname, batch_size=batch_size, num_epochs=num_epochs)
    input_labels = tf.placeholder(tf.float32, [None, 13, 13, 5, 85])
    input_images = tf.placeholder(tf.float32, [None, 416, 416, 3])

    norm_imgs = (input_images - 172.5) / 255.0

    net_output = darknet(norm_imgs, n_last_channels=425)

    prediction = lossfunction.output_to_prediction(darknet_output=net_output)

    loss = lossfunction.compute_loss(net_output, input_labels)

    coco = COCO(annotationFile)

    global_step = tf.Variable(0, trainable=False)
    decay_learning_rate = tf.train.exponential_decay(0.0001, global_step=global_step, decay_steps=1000, decay_rate=0.9)
    train_step = tf.train.AdamOptimizer(decay_learning_rate).minimize(loss, global_step=global_step)

    #saver = tf.train.import_meta_graph('./model/yolo2_coco.ckpt.meta')
    saver = tf.train.Saver()

    if istrain:
        with tf.Session() as sess:
            print('Start training...')
            start_time = time.time()
            sess.run(tf.global_variables_initializer())
            #saver.restore(sess=sess, save_path='./model/yolo2_coco.ckpt')

            try:
                for epoch in range(num_epochs):
                    epochtime = time.time()
                    for step in range(int(image_numbers/batch_size)):

                        timages_batch, tids_batch, tshapes_batch = sess.run([images_batch, ids_batch, shapes_batch])
                        labels = dataset.get_labels(images_ids=tids_batch, coco=coco)

                        train_step.run(feed_dict={input_images:timages_batch, input_labels: labels})
                        #train_step.run(feed_dict={input_labels: labels})

                        if step % 100 == 0:
                            tloss, tpreds = sess.run([loss, prediction], feed_dict={input_images:timages_batch, input_labels: labels})
                            print('Epoch {:>2}/{}, step = {:>6}/{:>6}, loss = {:.6f}, time = {}'
                                  .format(epoch, num_epochs, step, int(image_numbers / batch_size), tloss,
                                          time.time() - start_time))

                            for i in range(len(timages_batch)):
                                timage = np.uint8(timages_batch[i])

                                tpred = tpreds[i]
                                tshape = [tshapes_batch[0][i], tshapes_batch[1][i]]

                                resultimg = dataset.decode_labels(timage, tpred, tshape, sess)
                                cv2.imwrite(img_folder + 'epoch{}_step{}_i{}.png'.format(epoch, step, i), resultimg)
                                if i>=2:
                                    break
                                #cv2.imshow('testimg', resultimg)
                                #cv2.waitKey(2500)
                                #cv2.destroyAllWindows()
                                #break

                            print('-------------------------------------------------------------------------------------')
                    if epoch % save_epoch_number == 0:
                        saver.save(sess, model_folder + 'epoch{}.ckpt'.format(epoch), global_step=global_step)
                        print('Model saved in: {}'.format(model_folder + 'epoch{}.ckpt'.format(epoch)))
            except tf.errors.OutOfRangeError:
                print('End training...')
            finally:
                total_time = time.time() - start_time
                saver.save(sess, modelfile, global_step=global_step)
                print('Model saved as: {}, runing time: {} s'.format(modelfile, total_time))
                print('Done!')
    else:
        with tf.Session() as sess:
            print('Start testing...')
            sess.run(tf.global_variables_initializer())
            saver.restore(sess=sess, save_path=modelfile)

            for step in range(int(image_numbers / batch_size)):
                print('step: {}'.format(step))

                timages_batch, tids_batch, tshapes_batch = sess.run([images_batch, ids_batch, shapes_batch])

                tpreds = sess.run(prediction, feed_dict={input_images:timages_batch})

                for i in range(len(timages_batch)):
                    timage = np.uint8(timages_batch[i])

                    tpred = tpreds[i]
                    tshape = [tshapes_batch[0][i], tshapes_batch[1][i]]

                    resultimg = dataset.decode_labels(timage, tpred, tshape, sess)
                    cv2.imwrite(img_folder + 'step{}_i{}.png'.format(step, i), resultimg)

                    # cv2.imshow('testimg', resultimg)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
            print('Finished tesing. The test results are saved in {}.'.format(img_folder))

if __name__ == '__main__':
    _main()
