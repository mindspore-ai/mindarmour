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
Base Class of Attack.
"""
from abc import abstractmethod

import numpy as np

from mindarmour.utils._check_param import check_pair_numpy_param, \
    check_int_positive, check_equal_shape, check_numpy_param, check_model
from mindarmour.utils.util import calculate_iou
from mindarmour.utils.logger import LogUtil
from mindarmour.adv_robustness.attacks.black.black_model import BlackModel

LOGGER = LogUtil.get_instance()
TAG = 'Attack'


class Attack:
    """
    The abstract base class for all attack classes creating adversarial examples.
    """
    def __init__(self):
        pass

    def batch_generate(self, inputs, labels, batch_size=64):
        """
        Generate adversarial examples in batch, based on input samples and
        their labels.

        Args:
            inputs (Union[numpy.ndarray, tuple]): Samples based on which adversarial
                examples are generated.
            labels (Union[numpy.ndarray, tuple]): Original/target labels. \
                For each input if it has more than one label, it is wrapped in a tuple.
            batch_size (int): The number of samples in one batch.

        Returns:
            numpy.ndarray, generated adversarial examples

        Examples:
            >>> inputs = np.array([[0.2, 0.4, 0.5, 0.2], [0.7, 0.2, 0.4, 0.3]])
            >>> labels = np.array([3, 0])
            >>> advs = attack.batch_generate(inputs, labels, batch_size=2)
        """
        inputs_image = inputs[0] if isinstance(inputs, tuple) else inputs
        if isinstance(inputs, tuple):
            for i, inputs_item in enumerate(inputs):
                _ = check_pair_numpy_param('inputs_image', inputs_image, \
                    'inputs[{}]'.format(i), inputs_item)
        if isinstance(labels, tuple):
            for i, labels_item in enumerate(labels):
                _ = check_pair_numpy_param('inputs_image', inputs_image, \
                    'labels[{}]'.format(i), labels_item)
        else:
            _ = check_pair_numpy_param('inputs', inputs_image, \
                'labels', labels)
        arr_x = inputs
        arr_y = labels
        len_x = inputs_image.shape[0]
        batch_size = check_int_positive('batch_size', batch_size)
        batches = int(len_x / batch_size)
        rest = len_x - batches*batch_size
        res = []
        for i in range(batches):
            if isinstance(arr_x, tuple):
                x_batch = tuple([sub_items[i*batch_size: (i + 1)*batch_size] for sub_items in arr_x])
            else:
                x_batch = arr_x[i*batch_size: (i + 1)*batch_size]
            if isinstance(arr_y, tuple):
                y_batch = tuple([sub_labels[i*batch_size: (i + 1)*batch_size] for sub_labels in arr_y])
            else:
                y_batch = arr_y[i*batch_size: (i + 1)*batch_size]
            adv_x = self.generate(x_batch, y_batch)
            # Black-attack methods will return 3 values, just get the second.
            res.append(adv_x[1] if isinstance(adv_x, tuple) else adv_x)

        if rest != 0:
            if isinstance(arr_x, tuple):
                x_batch = tuple([sub_items[batches*batch_size:] for sub_items in arr_x])
            else:
                x_batch = arr_x[batches*batch_size:]
            if isinstance(arr_y, tuple):
                y_batch = tuple([sub_labels[batches*batch_size:] for sub_labels in arr_y])
            else:
                y_batch = arr_y[batches*batch_size:]
            adv_x = self.generate(x_batch, y_batch)
            # Black-attack methods will return 3 values, just get the second.
            res.append(adv_x[1] if isinstance(adv_x, tuple) else adv_x)

        adv_x = np.concatenate(res, axis=0)
        return adv_x

    @abstractmethod
    def generate(self, inputs, labels):
        """
        Generate adversarial examples based on normal samples and their labels.

        Args:
            inputs (Union[numpy.ndarray, tuple]): Samples based on which adversarial
                examples are generated.
            labels (Union[numpy.ndarray, tuple]): Original/target labels. \
                For each input if it has more than one label, it is wrapped in a tuple.

        Raises:
            NotImplementedError: It is an abstract method.
        """
        msg = 'The function generate() is an abstract function in class ' \
              '`Attack` and should be implemented in child class.'
        LOGGER.error(TAG, msg)
        raise NotImplementedError(msg)

    @staticmethod
    def _reduction(x_ori, q_times, label, best_position, model, targeted_attack):
        """
        Decrease the differences between the original samples and adversarial samples.

        Args:
            x_ori (numpy.ndarray): Original samples.
            q_times (int): Query times.
            label (int): Target label ot ground-truth label.
            best_position (numpy.ndarray): Adversarial examples.
            model (BlackModel): Target model.
            targeted_attack (bool): If True, it means this is a targeted attack. If False,
                it means this is an untargeted attack.

        Returns:
            numpy.ndarray, adversarial examples after reduction.

        Examples:
            >>> adv_reduction = self._reduction(self, [0.1, 0.2, 0.3], 20, 1,
            >>> [0.12, 0.15, 0.25])
        """
        LOGGER.info(TAG, 'Reduction begins...')
        model = check_model('model', model, BlackModel)
        x_ori = check_numpy_param('x_ori', x_ori)
        best_position = check_numpy_param('best_position', best_position)
        x_ori, best_position = check_equal_shape('x_ori', x_ori,
                                                 'best_position', best_position)
        x_ori_fla = x_ori.flatten()
        best_position_fla = best_position.flatten()
        pixel_deep = np.max(x_ori) - np.min(x_ori)
        nums_pixel = len(x_ori_fla)
        for i in range(nums_pixel):
            diff = x_ori_fla[i] - best_position_fla[i]
            if abs(diff) > pixel_deep*0.1:
                best_position_fla[i] += diff*0.5
                cur_label = np.argmax(
                    model.predict(best_position_fla.reshape(x_ori.shape)))
                q_times += 1
                if targeted_attack:
                    if cur_label != label:
                        best_position_fla[i] -= diff * 0.5

                else:
                    if cur_label == label:
                        best_position_fla -= diff*0.5
        return best_position_fla.reshape(x_ori.shape), q_times

    def _fast_reduction(self, x_ori, best_position, q_times, auxiliary_inputs, gt_boxes, gt_labels, model):
        """
        Decrease the differences between the original samples and adversarial samples in a fast way.

        Args:
            x_ori (numpy.ndarray): Original samples.
            best_position (numpy.ndarray): Adversarial examples.
            q_times (int): Query times.
            auxiliary_inputs (tuple): Auxiliary inputs mathced with x_ori.
            gt_boxes (numpy.ndarray): Ground-truth boxes of x_ori.
            gt_labels (numpy.ndarray): Ground-truth labels of x_ori.
            model (BlackModel): Target model.

        Returns:
            - numpy.ndarray, adversarial examples after reduction.

            - int, total query times after reduction.
        """
        LOGGER.info(TAG, 'Reduction begins...')
        model = check_model('model', model, BlackModel)
        x_ori = check_numpy_param('x_ori', x_ori)
        best_position = check_numpy_param('best_position', best_position)
        x_ori, best_position = check_equal_shape('x_ori', x_ori, 'best_position', best_position)
        x_shape = best_position.shape
        reduction_iters = 1000  # recover 0.1% each step
        _, original_num = self._detection_scores((best_position,) + auxiliary_inputs, gt_boxes, gt_labels, model)
        for _ in range(reduction_iters):
            diff = x_ori - best_position
            res = 0.5*diff*(np.random.random(x_shape) < 0.001)
            best_position += res
            _, correct_num = self._detection_scores((best_position,) + auxiliary_inputs, gt_boxes, gt_labels, model)
            q_times += 1
            if correct_num > original_num:
                best_position -= res
        return best_position, q_times

    @staticmethod
    def _detection_scores(inputs, gt_boxes, gt_labels, model):
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
        model = check_model('model', model, BlackModel)
        box_and_confi, pred_labels = model.predict(*inputs)
        det_scores = []
        correct_labels_num = []
        gt_boxes_num = gt_boxes.shape[0]
        iou_thres = 0.5
        for boxes, labels in zip(box_and_confi, pred_labels):
            score = 0
            box_num = boxes.shape[0]
            correct_label_flag = np.zeros(gt_labels.shape)
            for i in range(box_num):
                pred_box = boxes[i]
                max_iou_confi = 0
                for j in range(gt_boxes_num):
                    iou = calculate_iou(pred_box[:4], gt_boxes[j][:4])
                    if labels[i] == gt_labels[j] and iou > iou_thres:
                        max_iou_confi = max(max_iou_confi, pred_box[-1] + iou)
                        correct_label_flag[j] = 1
                score += max_iou_confi
            det_scores.append(score)
            correct_labels_num.append(np.sum(correct_label_flag))
        return np.array(det_scores), np.array(correct_labels_num)
