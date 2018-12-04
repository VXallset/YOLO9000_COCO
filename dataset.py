"""
By: Yu Sun
vxallset@outlook.com
Last modified: Dec 1st, 2018
All right reserved.
"""
from pycocotools.coco import COCO
import numpy as np
import tensorflow as tf
import cv2

# please install pycocotools at first


def IoU(box1_xy, box1_hw, box2_xy, box2_hw):
    """
    Calculate the IoU of two boxes (ndarrays)
    :param box1_xy: [..., 2] the x and y of the first box
    :param box1_hw: [..., 2] the height and width of the first box
    :param box2_xy: [..., 2] the x and y of the second box
    :param box2_hw: [..., 2] the height and width of the first box
    :return: [...] the iou of two boxes (ndarrays)
    """
    # box1_xy, box1_hw, box2_xy, box2_hw = np.array(box1_xy), np.array(box1_hw), np.array(box2_xy), np.array(box2_hw)

    # calculate the top-left and bottom-right point of the predicted box
    box1_xy_min, box1_xy_max = box1_xy - box1_hw/2.0, box1_xy + box1_hw/2.0

    # calculate the top-left and bottom-right point of the ground truth box
    box2_xy_min, box2_xy_max = box2_xy - box2_hw/2.0, box2_xy + box2_hw/2.0
    intersection_min = np.maximum(box2_xy_min, box1_xy_min)
    intersection_max = np.minimum(box2_xy_max, box1_xy_max)
    intersection_wh = np.maximum(intersection_max - intersection_min, 0.0)

    # calculate the intersection area and the union area of the prediction and the ground truth
    intersection_area = np.multiply(intersection_wh[..., 0], intersection_wh[..., 1])
    union_area = np.multiply(box2_hw[..., 0], box2_hw[..., 1]) + np.multiply(box1_hw[..., 0], box1_hw[..., 1]) - intersection_area
    iou = intersection_area / union_area
    return iou


def cococatID_2_mycatID(cococatid):
    """
    This function convert the id of an image in the COCO dataset (1~90, missing some cats) to my id (0~79)
    :param cocoid: int
    :return: myid: int
    """
    """
    coco = COCO('../dataset/COCO/annotations_trainval2017/annotations/instances_val2017.json')
    cats = coco.loadCats(coco.getCatIds())    
    mapdic = {}
    for i in range(len(cats)):
        mapdic[cats[i]['id']] = i
    """

    mapdic = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14,
              17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26,
              32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38,
              44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50,
              57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62,
              73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74,
              86: 75, 87: 76, 88: 77, 89: 78, 90: 79}
    myid = mapdic.get(cococatid)

    if myid != None:
        return myid
    else:
        print('Convert the coco ID to my ID failed! Given cococatid = {}'.format(cococatid))
        return -1


def myID_2_cococatname(myid):
    """
    This function convert my id to the name of the category in the coco dataset
    :param myid:
    :return:
    """
    """
    coco = COCO(annotationFile)

    cats = coco.loadCats(coco.getCatIds())
    dict = {}
    for i in range(len(cats)):
        dict[i] = cats[i]['name']
    print(dict)
    """
    mapdic = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
              8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
              14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
              22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
              29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
              35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
              40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
              47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
              54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
              61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
              68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
              75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    cococatname = mapdic.get(myid)

    if cococatname != None:
        return cococatname
    else:
        print('Convert my ID to cococatname failed! Given myid = {}'.format(myid))
        return -1


def decode_from_tfrecords(proto):
    """
    This function is used to decode a TFRecords file.

    :return: image, image_id, image_shape. image: an tf.float32 image. image_id: the file name of the image, image_shape: the shape
    of the image
    """
    features = tf.parse_single_example(proto,
                                       features={'image_id': tf.FixedLenFeature([], tf.int64),
                                                 'image_raw': tf.FixedLenFeature([], tf.string),
                                                 'height': tf.FixedLenFeature([], tf.int64),
                                                 'width': tf.FixedLenFeature([], tf.int64),
                                                 'channels': tf.FixedLenFeature([], tf.int64)})

    image = tf.image.decode_jpeg(features['image_raw'], channels=3)
    # set the shape of the output image to be [features['height'], features['width'], 3]
    image = tf.cast(tf.image.resize_images(image, (416, 416)), dtype=tf.float32)
    image.set_shape([416, 416, 3])

    # resize the image in the tfrecords. otherwise it will lead to some bugs when generate a batch of images, and cast
    # the datatype of the image to be float32, so that it can be passed to the network directly
    image_id = features['image_id']
    image_shape = (features['height'], features['width'], features['channels'])

    return image, image_id, image_shape


def input_batch(datasetname, batch_size, num_epochs):
    """
    This function is used to decode the TFrecord and return a batch of images as well as their information
    :param datasetname: the name of the TFrecord file.
    :param batch_size: the number of images in a batch
    :param num_epochs: the number of epochs
    :return: a batch of images as well as their information
    """
    with tf.name_scope('input_batch'):
        # The shuffle transformation uses a finite-sized buffer to shuffle elements
        # in memory. The parameter is the number of elements in the buffer. For
        # completely uniform shuffling, set the parameter to be the same as the
        # number of elements in the dataset.
        mydataset = tf.data.TFRecordDataset(datasetname)
        mydataset = mydataset.map(decode_from_tfrecords)

        # have no idea why I can't set the parameter of mydataset.shuffle to be the number of the dataset......
        #mydataset = mydataset.shuffle(200)
        mydataset = mydataset.repeat(num_epochs*2)
        # drop all the data that can't be used to make up a batch
        mydataset = mydataset.batch(batch_size, drop_remainder=True)
        iterator = mydataset.make_one_shot_iterator()

        nextelement = iterator.get_next()
        return nextelement


def get_labels(images_ids, coco):
    """
    This function is used to get the label of a batch of images.
    :param images_ids: int, the ids of a batch of images in the COCO dataset
    :param coco: the object which contains all information of the dataset
    :return: labels: [batch_size, 13, 13, 5, 5] np.float32 array, it contains the x, y (offset from the top-left of the
                     cell), height, width ( the ratio of the ground truth height and width with the box priors),
                     and the class id
             mask: [batch_size, 13, 13, 5] np.int64 array, used to mark which box needs to describe the ground truth
    """
    #coco = COCO(cocofile)

    cocoimgs = coco.loadImgs(images_ids)

    box_priors = np.array([[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]])

    labels = np.zeros((len(cocoimgs), 13, 13, 5, 85), dtype=np.float32)
    mask = np.zeros((len(cocoimgs), 13, 13, 5), dtype=np.int64)
    eps = 1e-50

    for i_img in range(len(cocoimgs)):
        cocoimg = cocoimgs[i_img]
        ann_ids = coco.getAnnIds(cocoimg['id'])
        annotations = coco.loadAnns(ann_ids)
        #print('{}th image:'.format(i_img), annotations[0]['image_id'])

        shape_x, shape_y = cocoimg['height'], cocoimg['width']

        # 'bbox': [a, b, width, height]
        for annotation in annotations:
            # if the annotation is crowd, continue

            if annotation['iscrowd']:
                continue

            bbox = annotation['bbox']
            cat_id = annotation['category_id']
            #convert the coco id to my id
            cat_id = cococatID_2_mycatID(cat_id)

            # the point in the original image times can be cast into the 13x13 feature map
            # by multiplying the rate_x and rate_y
            rate_x = 13.0 / shape_x
            rate_y = 13.0 / shape_y

            [b, a, width, height] = bbox
            ground_truth_center_x = (a + height / 2) * rate_x
            ground_truth_center_y = (b + width / 2) * rate_y
            ground_truth_height = height * rate_x
            ground_truth_width = width * rate_y

            box_center_x = np.floor(ground_truth_center_x) + 0.5
            box_center_y = np.floor(ground_truth_center_y) + 0.5

            iou_index = -1
            iou_max = -1

            for box_num in range(5):
                pw = box_priors[box_num][0]
                ph = box_priors[box_num][1]
                box_width = pw * rate_y
                box_height = ph * rate_x

                iou_temp = IoU(np.array([a*rate_x, b * rate_y]), np.array([height*rate_x, width *rate_y]),
                                     np.array([box_center_x-box_height/2, box_center_y - box_width/2]),
                                     np.array([box_height, box_width]))
                if iou_temp >= iou_max:
                    iou_max = iou_temp
                    iou_index = box_num
            if iou_index < 0 or iou_index >= 5:
                print('!!!!!!!!!!!!!!!!     Calaulate IOU error       !!!!!!!!!!!!!!!!')
            # all the values saved in the labels take the 1x1 feature map as the coordinate system
            labels[i_img, int(np.floor(ground_truth_center_x)), int(np.floor(ground_truth_center_y)), int(iou_index), :4] \
                = (ground_truth_center_x - np.floor(ground_truth_center_x)), \
                   (ground_truth_center_y - np.floor(ground_truth_center_y)), \
                  ground_truth_height / (box_priors[iou_index][0]), \
                  ground_truth_width / (box_priors[iou_index][1])
            # set the confidence to be 1
            labels[i_img, int(np.floor(ground_truth_center_x)), int(np.floor(ground_truth_center_y)), int(iou_index),
                   4] = 1
            # one-hot, set the class number
            labels[i_img, int(np.floor(ground_truth_center_x)), int(np.floor(ground_truth_center_y)), int(iou_index),
                   cat_id + 5] = 1
    return labels


def draw_boxes_on_image(input_img, boxes):
    """
    Draw boxes on the original input image.
    :param input_img: input image
    :param boxes: [] box coordinate, boxes[..., :2]: top-left point of the box, boxes[..., 2:4]: bottom-right of the box
    :return:
    """
    """
    all_class = [{'id': 1, 'class_name': 'person'}, {'id': 2, 'class_name': 'bicycle'}, {'id': 3, 'class_name': 'car'},
                 {'id': 4, 'class_name': 'motorcycle'}, {'id': 5, 'class_name': 'airplane'},
                 {'id': 6, 'class_name': 'bus'}, {'id': 7, 'class_name': 'train'}, {'id': 8, 'class_name': 'truck'},
                 {'id': 9, 'class_name': 'boat'}, {'id': 10, 'class_name': 'traffic light'},
                 {'id': 11, 'class_name': 'fire hydrant'}, {'id': 12, 'class_name': 'Unknown_12'},
                 {'id': 13, 'class_name': 'stop sign'}, {'id': 14, 'class_name': 'parking meter'},
                 {'id': 15, 'class_name': 'bench'}, {'id': 16, 'class_name': 'bird'}, {'id': 17, 'class_name': 'cat'},
                 {'id': 18, 'class_name': 'dog'}, {'id': 19, 'class_name': 'horse'}, {'id': 20, 'class_name': 'sheep'},
                 {'id': 21, 'class_name': 'cow'}, {'id': 22, 'class_name': 'elephant'},
                 {'id': 23, 'class_name': 'bear'}, {'id': 24, 'class_name': 'zebra'},
                 {'id': 25, 'class_name': 'giraffe'}, {'id': 26, 'class_name': 'Unknown_26'},
                 {'id': 27, 'class_name': 'backpack'}, {'id': 28, 'class_name': 'umbrella'},
                 {'id': 29, 'class_name': 'Unknown_29'}, {'id': 31, 'class_name': 'handbag'},
                 {'id': 31, 'class_name': 'Unknown_31'}, {'id': 32, 'class_name': 'tie'},
                 {'id': 33, 'class_name': 'suitcase'}, {'id': 34, 'class_name': 'frisbee'},
                 {'id': 35, 'class_name': 'skis'}, {'id': 36, 'class_name': 'snowboard'},
                 {'id': 37, 'class_name': 'sports ball'}, {'id': 38, 'class_name': 'kite'},
                 {'id': 39, 'class_name': 'baseball bat'}, {'id': 40, 'class_name': 'baseball glove'},
                 {'id': 41, 'class_name': 'skateboard'}, {'id': 42, 'class_name': 'surfboard'},
                 {'id': 43, 'class_name': 'tennis racket'}, {'id': 44, 'class_name': 'bottle'},
                 {'id': 45, 'class_name': 'Unknown_45'}, {'id': 46, 'class_name': 'wine glass'},
                 {'id': 47, 'class_name': 'cup'}, {'id': 48, 'class_name': 'fork'}, {'id': 49, 'class_name': 'knife'},
                 {'id': 50, 'class_name': 'spoon'}, {'id': 51, 'class_name': 'bowl'},
                 {'id': 52, 'class_name': 'banana'}, {'id': 53, 'class_name': 'apple'},
                 {'id': 54, 'class_name': 'sandwich'}, {'id': 55, 'class_name': 'orange'},
                 {'id': 56, 'class_name': 'broccoli'}, {'id': 57, 'class_name': 'carrot'},
                 {'id': 58, 'class_name': 'hot dog'}, {'id': 59, 'class_name': 'pizza'},
                 {'id': 60, 'class_name': 'donut'}, {'id': 61, 'class_name': 'cake'}, {'id': 62, 'class_name': 'chair'},
                 {'id': 63, 'class_name': 'couch'}, {'id': 64, 'class_name': 'potted plant'},
                 {'id': 65, 'class_name': 'bed'}, {'id': 66, 'class_name': 'Unknown_66'},
                 {'id': 67, 'class_name': 'dining table'}, {'id': 68, 'class_name': 'Unknown_68'},
                 {'id': 70, 'class_name': 'toilet'}, {'id': 70, 'class_name': 'Unknown_70'},
                 {'id': 72, 'class_name': 'tv'}, {'id': 72, 'class_name': 'Unknown_72'},
                 {'id': 73, 'class_name': 'laptop'}, {'id': 74, 'class_name': 'mouse'},
                 {'id': 75, 'class_name': 'remote'}, {'id': 76, 'class_name': 'keyboard'},
                 {'id': 77, 'class_name': 'cell phone'}, {'id': 78, 'class_name': 'microwave'},
                 {'id': 79, 'class_name': 'oven'}, {'id': 80, 'class_name': 'toaster'},
                 {'id': 81, 'class_name': 'sink'}, {'id': 82, 'class_name': 'refrigerator'},
                 {'id': 83, 'class_name': 'Unknown_83'}, {'id': 84, 'class_name': 'book'},
                 {'id': 85, 'class_name': 'clock'}, {'id': 86, 'class_name': 'vase'},
                 {'id': 87, 'class_name': 'scissors'}, {'id': 88, 'class_name': 'teddy bear'},
                 {'id': 89, 'class_name': 'hair drier'}, {'id': 90, 'class_name': 'toothbrush'}]
    """

    for box in boxes:
        #print('box = {}'.format(box))
        x1, y1, x2, y2, class_id = box
        minimum = -999999
        if x1 < minimum or y1 < minimum or x2 < minimum or y2 < minimum:
            continue
        classname = myID_2_cococatname(class_id)
        cv2.rectangle(input_img, (y1, x1), (y2, x2), color=[255, 0, 0])
        cv2.putText(input_img, '{}'.format(classname), (y1, x1 - 5),
                    fontFace=cv2.FONT_ITALIC, fontScale=0.6, color=[255, 0, 0])
    return input_img


def decode_labels(image, label, input_img_shapes, sess_tmp, threshold=0.3):
    """
    Convert the prediction labels to the boxes and draw them on the original image
    :param image: 416 x 416 x 3 image
    :param label: 13 x 13 x 425 float32 np array
    :param input_img_shapes: [height, weight]
    :param threshold: confidence threshold, remove all the boxes whose confidences are smaller than threshold.
    :return: the image with rectangles, category names
    """
    image = cv2.resize(image, (input_img_shapes[1], input_img_shapes[0]))
    box_priors = np.array([[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]])
    obj_prob = label[..., 4]
    # print(obj_prob[:2, 0, :])
    # print('the max of conf:{}'.format(np.max(obj_prob)))
    class_prob = label[..., 5:]
    # class_prob = np.max(class_prob, -1)
    # print('the max of class prob:{}'.format(np.max(class_prob)))

    mask = label[..., 4] > threshold
    boxes = label[mask]

    box_number = np.sum(np.int64(mask))

    if box_number >= 0:
        biases = np.array([[[[i, j, k] for k in range(5)] for j in range(13)] for i in range(13)])
        # biases: [box_number, 3], for each box, the coordinate information is [cell_x, cell_y, in_which_cell]
        biases = biases[mask]

        # convert the box coordinate from the feature map (1x1) to be the real coordinate (height, width),
        # define the x-axis direction is down and y-axis direction is right
        rate_x = input_img_shapes[0] / 13.0
        rate_y = input_img_shapes[1] / 13.0

        original_coordinates = np.zeros((box_number, 5), dtype=np.int64)
        for box_i in range(box_number):
            x = (boxes[box_i][0] + biases[box_i, 0]) * rate_x
            y = (boxes[box_i][1] + biases[box_i, 1]) * rate_y
            higth = boxes[box_i][2] * box_priors[biases[box_i, 2], 0] * rate_x
            width = boxes[box_i][3] * box_priors[biases[box_i, 2], 1] * rate_y

            x1 = np.int64(x - higth / 2.0)
            y1 = np.int64(y - width / 2.0)
            x2 = np.int64(x + higth / 2.0)
            y2 = np.int64(y + width / 2.0)
            # here, the last dim should be the class id, start from 1
            original_coordinates[box_i] = [x1, y1, x2, y2, np.argmax(boxes[box_i][5:])]

        box_coor = original_coordinates[..., :4]
        box_score = boxes[..., 4]

        selected_indeces = sess_tmp.run(tf.image.non_max_suppression(box_coor, box_score, max_output_size=3))
        selected_boxes = sess_tmp.run(tf.gather(original_coordinates, selected_indeces))
        image = draw_boxes_on_image(image, selected_boxes)
    image = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2BGR)
    return image
