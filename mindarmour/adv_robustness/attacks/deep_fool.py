# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DeepFool Attack.
"""
import numpy as np

from mindspore import Tensor
from mindspore.nn import Cell

from mindarmour.utils.logger import LogUtil
from mindarmour.utils.util import GradWrap, jacobian_matrix, \
    jacobian_matrix_for_detection, calculate_iou, to_tensor_tuple
from mindarmour.utils._check_param import check_pair_numpy_param, check_model, \
    check_value_positive, check_int_positive, check_norm_level, \
    check_param_multi_types, check_param_type, check_value_non_negative
from .attack import Attack

LOGGER = LogUtil.get_instance()
TAG = 'DeepFool'


class _GetLogits(Cell):
    def __init__(self, network):
        super(_GetLogits, self).__init__()
        self._network = network

    def construct(self, *inputs):
        _, pre_logits = self._network(*inputs)
        return pre_logits


def _deepfool_detection_scores(inputs, gt_boxes, gt_labels, network):
    """
    Evaluate the detection result of inputs, specially for object detection models.

    Args:
        inputs (numpy.ndarray): Input samples.
        gt_boxes (numpy.ndarray): Ground-truth boxes of inputs.
        gt_labels (numpy.ndarray): Ground-truth labels of inputs.
        model (BlackModel): Target model.

    Returns:
        - numpy.ndarray, detection scores of inputs.

        - numpy.ndarray, the number of objects that are correctly detected.
    """
    network = check_param_type('network', network, Cell)
    inputs_tensor = to_tensor_tuple(inputs)
    box_and_confi, pred_logits = network(*inputs_tensor)
    box_and_confi, pred_logits = box_and_confi.asnumpy(), pred_logits.asnumpy()
    pred_labels = np.argmax(pred_logits, axis=2)
    det_scores = []
    correct_labels_num = []
    gt_boxes_num = gt_boxes.shape[1]
    iou_thres = 0.5
    for idx, (boxes, labels) in enumerate(zip(box_and_confi, pred_labels)):
        score = 0
        box_num = boxes.shape[0]
        gt_boxes_idx = gt_boxes[idx]
        gt_labels_idx = gt_labels[idx]
        correct_label_flag = np.zeros(gt_labels_idx.shape)
        for i in range(box_num):
            pred_box = boxes[i]
            max_iou_confi = 0
            for j in range(gt_boxes_num):
                iou = calculate_iou(pred_box[:4], gt_boxes_idx[j][:4])
                if labels[i] == gt_labels_idx[j] and iou > iou_thres:
                    max_iou_confi = max(max_iou_confi, pred_box[-1] + iou)
                    correct_label_flag[j] = 1
            score += max_iou_confi
        det_scores.append(score)
        correct_labels_num.append(np.sum(correct_label_flag))
    return np.array(det_scores), np.array(correct_labels_num)


def _is_success(inputs, gt_boxes, gt_labels, network, gt_object_nums, reserve_ratio):
    _, correct_nums_adv = _deepfool_detection_scores(inputs, gt_boxes, gt_labels, network)
    return np.all(correct_nums_adv <= (gt_object_nums*reserve_ratio).astype(np.int32))


class DeepFool(Attack):
    """
    DeepFool is an untargeted & iterative attack achieved by moving the benign
    sample to the nearest classification boundary and crossing the boundary.

    Reference: `DeepFool: a simple and accurate method to fool deep neural
    networks <https://arxiv.org/abs/1511.04599>`_

    Args:
        network (Cell): Target model.
        num_classes (int): Number of labels of model output, which should be
            greater than zero.
        model_type (str): Tye type of targeted model. 'classification' and 'detection' are supported now.
            default: 'classification'.
        reserve_ratio (Union[int, float]): The percentage of objects that can be detected after attaks,
            specifically for model_type='detection'. Reserve_ratio should be in the range of (0, 1). Default: 0.3.
        max_iters (int): Max iterations, which should be
            greater than zero. Default: 50.
        overshoot (float): Overshoot parameter. Default: 0.02.
        norm_level (Union[int, str]): Order of the vector norm. Possible values: np.inf
            or 2. Default: 2.
        bounds (Union[tuple, list]): Upper and lower bounds of data range. In form of (clip_min,
            clip_max). Default: None.
        sparse (bool): If True, input labels are sparse-coded. If False,
            input labels are onehot-coded. Default: True.

    Examples:
        >>> attack = DeepFool(network)
    """

    def __init__(self, network, num_classes, model_type='classification',
                 reserve_ratio=0.3, max_iters=50, overshoot=0.02, norm_level=2, bounds=None, sparse=True):
        super(DeepFool, self).__init__()
        self._network = check_model('network', network, Cell)
        self._max_iters = check_int_positive('max_iters', max_iters)
        self._overshoot = check_value_positive('overshoot', overshoot)
        self._norm_level = check_norm_level(norm_level)
        self._num_classes = check_int_positive('num_classes', num_classes)
        self._net_grad = GradWrap(self._network)
        self._bounds = bounds
        if self._bounds is not None:
            self._bounds = check_param_multi_types('bounds', bounds, [list, tuple])
            for b in self._bounds:
                _ = check_param_multi_types('bound', b, [int, float])
        self._sparse = check_param_type('sparse', sparse, bool)
        self._model_type = check_param_type('model_type', model_type, str)
        if self._model_type not in ('classification', 'detection'):
            msg = "Only 'classification' or 'detection' is supported now, but got {}.".format(self._model_type)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        self._reserve_ratio = check_value_non_negative('reserve_ratio', reserve_ratio)
        if self._reserve_ratio > 1:
            msg = 'reserve_ratio should be less than 1.0, but got {}.'.format(self._reserve_ratio)
            LOGGER.error(TAG, msg)
            raise ValueError(TAG, msg)

    def generate(self, inputs, labels):
        """
        Generate adversarial examples based on input samples and original labels.

        Args:
            inputs (Union[numpy.ndarray, tuple]): Input samples. The format of inputs should be numpy.ndarray if
                model_type='classification'. The format of inputs can be (input1, input2, ...) or only one array if
                model_type='detection'.
            labels (Union[numpy.ndarray, tuple]): Targeted labels or ground-truth labels. The format of labels should
                be numpy.ndarray if model_type='classification'. The format of labels should be (gt_boxes, gt_labels)
                if model_type='detection'.

        Returns:
            numpy.ndarray, adversarial examples.

        Raises:
            NotImplementedError: If norm_level is not in [2, np.inf, '2', 'inf'].

        Examples:
            >>> advs = generate([[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]], [1, 2])
        """
        if self._model_type == 'detection':
            return self._generate_detection(inputs, labels)
        if self._model_type == 'classification':
            return self._generate_classification(inputs, labels)
        return None


    def _generate_detection(self, inputs, labels):
        """Generate adversarial examples in detection scenario"""
        images, auxiliary_inputs = inputs[0], inputs[1:]
        gt_boxes, gt_labels = labels
        _, gt_object_nums = _deepfool_detection_scores(inputs, gt_boxes, gt_labels, self._network)
        if not self._sparse:
            gt_labels = np.argmax(gt_labels, axis=2)
        origin_labels = np.zeros(gt_labels.shape[0])
        for i in range(gt_labels.shape[0]):
            origin_labels[i] = np.argmax(np.bincount(gt_labels[i]))
        images_dtype = images.dtype
        iteration = 0
        num_boxes = gt_labels.shape[1]
        merge_net = _GetLogits(self._network)
        detection_net_grad = GradWrap(merge_net)
        weight = np.squeeze(np.zeros(images.shape[1:]))
        r_tot = np.zeros(images.shape)
        x_origin = images
        while not _is_success((images,) + auxiliary_inputs, gt_boxes, gt_labels, self._network, gt_object_nums, \
                         self._reserve_ratio) and iteration < self._max_iters:
            preds_logits = merge_net(*to_tensor_tuple(images), *to_tensor_tuple(auxiliary_inputs)).asnumpy()
            grads = jacobian_matrix_for_detection(detection_net_grad, (images,) + auxiliary_inputs,
                                                  num_boxes, self._num_classes)
            for idx in range(images.shape[0]):
                diff_w = np.inf
                label = int(origin_labels[idx])
                auxiliary_input_i = tuple()
                for item in auxiliary_inputs:
                    auxiliary_input_i += (np.expand_dims(item[idx], axis=0),)
                gt_boxes_i = np.expand_dims(gt_boxes[idx], axis=0)
                gt_labels_i = np.expand_dims(gt_labels[idx], axis=0)
                inputs_i = (np.expand_dims(images[idx], axis=0),) + auxiliary_input_i
                if _is_success(inputs_i, gt_boxes_i, gt_labels_i,
                               self._network, gt_object_nums[idx], self._reserve_ratio):
                    continue
                for k in range(self._num_classes):
                    if k == label:
                        continue
                    w_k = grads[k, idx, ...] - grads[label, idx, ...]
                    f_k = np.mean(np.abs(preds_logits[idx, :, k, ...] - preds_logits[idx, :, label, ...]))
                    if self._norm_level == 2 or self._norm_level == '2':
                        diff_w_k = abs(f_k) / (np.linalg.norm(w_k) + 1e-8)
                    elif self._norm_level == np.inf \
                            or self._norm_level == 'inf':
                        diff_w_k = abs(f_k) / (np.linalg.norm(w_k, ord=1) + 1e-8)
                    else:
                        msg = 'ord {} is not available.' \
                            .format(str(self._norm_level))
                        LOGGER.error(TAG, msg)
                        raise NotImplementedError(msg)
                    if diff_w_k < diff_w:
                        diff_w = diff_w_k
                        weight = w_k
                if self._norm_level == 2 or self._norm_level == '2':
                    r_i = diff_w*weight / (np.linalg.norm(weight) + 1e-8)
                elif self._norm_level == np.inf or self._norm_level == 'inf':
                    r_i = diff_w*np.sign(weight) \
                          / (np.linalg.norm(weight, ord=1) + 1e-8)
                else:
                    msg = 'ord {} is not available in normalization,' \
                        .format(str(self._norm_level))
                    LOGGER.error(TAG, msg)
                    raise NotImplementedError(msg)
                r_tot[idx, ...] = r_tot[idx, ...] + r_i

            if self._bounds is not None:
                clip_min, clip_max = self._bounds
                images = x_origin + (1 + self._overshoot)*r_tot*(clip_max-clip_min)
                images = np.clip(images, clip_min, clip_max)
            else:
                images = x_origin + (1 + self._overshoot)*r_tot
            iteration += 1
            images = images.astype(images_dtype)
            del preds_logits, grads
        return images



    def _generate_classification(self, inputs, labels):
        """Generate adversarial examples in classification scenario"""
        inputs, labels = check_pair_numpy_param('inputs', inputs,
                                                'labels', labels)
        if not self._sparse:
            labels = np.argmax(labels, axis=1)
        inputs_dtype = inputs.dtype
        iteration = 0
        origin_labels = labels
        cur_labels = origin_labels.copy()
        weight = np.squeeze(np.zeros(inputs.shape[1:]))
        r_tot = np.zeros(inputs.shape)
        x_origin = inputs
        while np.any(cur_labels == origin_labels) and iteration < self._max_iters:
            preds = self._network(Tensor(inputs)).asnumpy()
            grads = jacobian_matrix(self._net_grad, inputs, self._num_classes)
            for idx in range(inputs.shape[0]):
                diff_w = np.inf
                label = origin_labels[idx]
                if cur_labels[idx] != label:
                    continue
                for k in range(self._num_classes):
                    if k == label:
                        continue
                    w_k = grads[k, idx, ...] - grads[label, idx, ...]
                    f_k = preds[idx, k] - preds[idx, label]
                    if self._norm_level == 2 or self._norm_level == '2':
                        diff_w_k = abs(f_k) / (np.linalg.norm(w_k) + 1e-8)
                    elif self._norm_level == np.inf \
                            or self._norm_level == 'inf':
                        diff_w_k = abs(f_k) / (np.linalg.norm(w_k, ord=1) + 1e-8)
                    else:
                        msg = 'ord {} is not available.' \
                            .format(str(self._norm_level))
                        LOGGER.error(TAG, msg)
                        raise NotImplementedError(msg)
                    if diff_w_k < diff_w:
                        diff_w = diff_w_k
                        weight = w_k

                if self._norm_level == 2 or self._norm_level == '2':
                    r_i = diff_w*weight / (np.linalg.norm(weight) + 1e-8)
                elif self._norm_level == np.inf or self._norm_level == 'inf':
                    r_i = diff_w*np.sign(weight) \
                          / (np.linalg.norm(weight, ord=1) + 1e-8)
                else:
                    msg = 'ord {} is not available in normalization.' \
                        .format(str(self._norm_level))
                    LOGGER.error(TAG, msg)
                    raise NotImplementedError(msg)
                r_tot[idx, ...] = r_tot[idx, ...] + r_i

            if self._bounds is not None:
                clip_min, clip_max = self._bounds
                inputs = x_origin + (1 + self._overshoot)*r_tot*(clip_max
                                                                 - clip_min)
                inputs = np.clip(inputs, clip_min, clip_max)
            else:
                inputs = x_origin + (1 + self._overshoot)*r_tot
            cur_labels = np.argmax(
                self._network(Tensor(inputs.astype(inputs_dtype))).asnumpy(),
                axis=1)
            iteration += 1
            inputs = inputs.astype(inputs_dtype)
            del preds, grads
        return inputs
