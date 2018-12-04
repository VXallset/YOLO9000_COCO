"""
By: Yu Sun
vxallset@outlook.com
Last modified: Dec 1st, 2018
All right reserved.
"""
import tensorflow as tf
import numpy as np


def output_to_prediction(darknet_output):
    """
    Converting the darknet output to the prediction values.
    :param darknet_output: darknet_output: [16, 13, 13, 425] tf.float32 tensor, the output of the darknet output
    :return: [16, 13, 13, 5, 85] tf.float32 tensor, the prediction labels of a batch of images
    """
    darknet_output = tf.reshape(darknet_output, [-1, 13, 13, 5, 85])

    xy_offset = tf.nn.sigmoid(darknet_output[..., 0:2])  # 中心坐标相对于该cell左上角的偏移量，sigmoid函数归一化到0-1

    wh_offset = tf.exp(darknet_output[..., 2:4])  # 相对于anchor的wh比例，通过e指数解码

    obj_probs = tf.nn.sigmoid(darknet_output[..., 4:5])  # 置信度，sigmoid函数归一化到0-1

    class_probs = tf.nn.softmax(darknet_output[..., 5:])  # 网络回归的是'得分',用softmax转变成类别概率

    prediction = tf.concat([xy_offset, wh_offset, obj_probs, class_probs], axis=-1)
    return prediction


def compute_loss(darknet_output, ground_truth):
    """
    Conpute the loss using the network output and the ground truth
    :param darknet_output: [16, 13, 13, 425] tf.float32 tensor, the output of the darknet output
    :param ground_truth: [16, 13, 13, 5, 85] tf.float32 tensor, the ground truth labels of a batch of images
    :return: float32 loss
    """
    darknet_output = tf.reshape(darknet_output, [-1, 13, 13, 5, 85])
    net_raw_xy = darknet_output[..., :2]
    net_raw_hw = darknet_output[..., 2:4]
    net_raw_conf = darknet_output[..., 4:5]
    net_raw_prob = darknet_output[..., 5:]


    prediction = output_to_prediction(darknet_output)
    prediction = tf.reshape(prediction, [-1, 13, 13, 5, 85])

    # the factor used to calculate the object weight
    obj_scale = 5

    # the factor used to calculate the no object weight
    no_obj_scale = 1

    # the factor used to calculate the class prediction loss
    class_scale = 1

    # the factor factor used to calculate the coordinate loss
    coordinates_scale = 1

    # decode the prediction, convert all values to the 13x13 feature map
    pred_xy_offset = prediction[..., :2]
    pred_hw_ratio = prediction[..., 2:4]
    pred_conf = prediction[..., 4:5]
    pred_class = prediction[..., 5:]

    # decode the ground truth, convert all values to the 13x13 feature map
    gt_xy_offset = ground_truth[..., :2]
    gt_hw_ratio = ground_truth[..., 2:4]
    gt_conf = ground_truth[..., 4:5]
    gt_class = ground_truth[..., 5:]

    # 13 x 13 x 2 tensor, used to compute the x and y in the 13 x 13 feature map
    biases = tf.Variable([[[j * 1.0, i * 1.0] for i in range(13)] for j in range(13)])
    biases = tf.reshape(biases, [1, 13, 13, 1, 2])

    box_priors = tf.Variable([[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]])
    box_priors = tf.reshape(box_priors, [1, 1, 1, 5, 2])

    pred_xy = pred_xy_offset + biases
    pred_hw = pred_hw_ratio * box_priors

    gt_xy = gt_xy_offset + biases
    gt_hw = gt_hw_ratio * box_priors

    # calculate the top-left and bottom-right point of the predicted box
    pred_xy_min, pred_xy_max = pred_xy - pred_hw / 2.0, pred_xy + pred_hw / 2.0

    gt_xy_min, gt_xy_max = gt_xy - gt_hw / 2.0, gt_xy + gt_hw / 2.0

    intersection_min = tf.maximum(gt_xy_min, pred_xy_min)
    intersection_max = tf.minimum(gt_xy_max, pred_xy_max)
    intersection_hw = tf.maximum(intersection_max - intersection_min, 0.0)

    # calculate the intersection area and the union area of the prediction and the ground truth
    intersection_area = tf.multiply(intersection_hw[..., 0], intersection_hw[..., 1])
    union_area = tf.multiply(gt_hw[..., 0], gt_hw[..., 1]) + tf.multiply(pred_hw[..., 0],
                                                                         pred_hw[..., 1]) - intersection_area
    # shape of iou: (?, 13, 13, 5)
    box_iou = intersection_area / union_area

    obj = gt_conf

    gt_raw_hw = tf.log(gt_hw_ratio)

    gt_raw_hw = tf.where(tf.is_inf(gt_raw_hw), tf.zeros_like(gt_raw_hw), gt_raw_hw)

    # ======================================================================================

    coords_xy_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=net_raw_xy, labels=gt_xy_offset) * obj * coordinates_scale
    coords_xy_loss = tf.reduce_sum(coords_xy_loss)

    coords_wh_loss = tf.square(net_raw_hw - gt_raw_hw) * 0.5 * obj * coordinates_scale

    coords_wh_loss = tf.reduce_sum(coords_wh_loss)

    coords_loss = coords_xy_loss + coords_wh_loss

    ignore_thresh = 0.5

    ignore_mask = tf.cast(tf.less(box_iou, ignore_thresh * tf.ones_like(box_iou)), tf.float32)
    ignore_mask = tf.reshape(ignore_mask, [-1, 13, 13, 5])
    ignore_mask = tf.expand_dims(ignore_mask, -1)

    back_loss = ((1 - obj) * tf.nn.sigmoid_cross_entropy_with_logits(logits=net_raw_conf, labels=obj) * ignore_mask)
    back_loss = tf.reduce_sum(back_loss)

    fore_loss = obj * tf.nn.sigmoid_cross_entropy_with_logits(logits=net_raw_conf, labels=obj)

    fore_loss = tf.reduce_sum(fore_loss)

    conf_loss = back_loss + fore_loss

    class_loss = tf.reduce_sum(obj * tf.nn.sigmoid_cross_entropy_with_logits(logits=net_raw_prob, labels=gt_class))

    loss = coords_loss + conf_loss + class_loss

    return loss
