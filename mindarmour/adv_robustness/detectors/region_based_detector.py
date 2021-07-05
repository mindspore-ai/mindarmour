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
Region-Based detector
"""
import time

import numpy as np

from mindspore import Model
from mindspore import Tensor

from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_numpy_param, check_param_type, \
    check_pair_numpy_param, check_model, check_int_positive, \
    check_value_positive, check_value_non_negative, check_param_in_range, \
    check_equal_shape
from .detector import Detector

LOGGER = LogUtil.get_instance()
TAG = 'RegionBasedDetector'


class RegionBasedDetector(Detector):
    """
    This class implement a region-based detector.

    Reference: `Mitigating evasion attacks to deep neural networks via
    region-based classification <https://arxiv.org/abs/1709.05583>`_

    Args:
        model (Model): Target model.
        number_points (int): The number of samples generate from the
            hyper cube of original sample. Default: 10.
        initial_radius (float): Initial radius of hyper cube. Default: 0.0.
        max_radius (float): Maximum radius of hyper cube. Default: 1.0.
        search_step (float): Incremental during search of radius. Default: 0.01.
        degrade_limit (float): Acceptable decrease of classification accuracy.
            Default: 0.0.
        sparse (bool): If True, input labels are sparse-encoded. If False,
            input labels are one-hot-encoded. Default: False.

    Examples:
        >>> detector = RegionBasedDetector(model)
        >>> detector.fit(ori, labels)
        >>> adv_ids = detector.detect(adv)
    """

    def __init__(self, model, number_points=10, initial_radius=0.0,
                 max_radius=1.0, search_step=0.01, degrade_limit=0.0,
                 sparse=False):
        super(RegionBasedDetector, self).__init__()
        self._model = check_model('targeted model', model, Model)
        self._number_points = check_int_positive('number_points', number_points)
        self._initial_radius = check_value_non_negative('initial_radius',
                                                        initial_radius)
        self._max_radius = check_value_positive('max_radius', max_radius)
        self._search_step = check_value_positive('search_step', search_step)
        self._degrade_limit = check_value_non_negative('degrade_limit',
                                                       degrade_limit)
        self._sparse = check_param_type('sparse', sparse, bool)
        self._radius = None

    def set_radius(self, radius):
        """
        Set radius.

        Args:
            radius (float): Radius of region.
        """
        self._radius = check_param_in_range('radius', radius,
                                            self._initial_radius,
                                            self._max_radius)

    def fit(self, inputs, labels=None):
        """
        Train detector to decide the best radius.

        Args:
            inputs (numpy.ndarray): Benign samples.
            labels (numpy.ndarray): Ground truth labels of the input samples.
                Default:None.

        Returns:
            float, the best radius.
        """
        inputs, labels = check_pair_numpy_param('inputs', inputs,
                                                'labels', labels)
        LOGGER.debug(TAG, 'enter fit() function.')
        time_start = time.time()
        search_iters = (self._max_radius
                        - self._initial_radius) / self._search_step
        search_iters = np.round(search_iters).astype(int)
        radius = self._initial_radius
        pred = self._model.predict(Tensor(inputs))
        raw_preds = np.argmax(pred.asnumpy(), axis=1)
        if not self._sparse:
            labels = np.argmax(labels, axis=1)
        raw_preds, labels = check_equal_shape('raw_preds', raw_preds, 'labels',
                                              labels)
        raw_acc = np.sum(raw_preds == labels) / inputs.shape[0]

        for _ in range(search_iters):
            rc_preds = self._rc_forward(inputs, radius)
            rc_preds, labels = check_equal_shape('rc_preds', rc_preds, 'labels',
                                                 labels)
            def_acc = np.sum(rc_preds == labels) / inputs.shape[0]
            if def_acc >= raw_acc - self._degrade_limit:
                radius += self._search_step
                continue
            break

        self._radius = radius - self._search_step
        LOGGER.debug(TAG, 'best radius is: %s', self._radius)
        LOGGER.debug(TAG,
                     'time used to train detector of %d samples is: %s seconds',
                     inputs.shape[0],
                     time.time() - time_start)
        return self._radius

    def _generate_hyper_cube(self, inputs, radius):
        """
        Generate random samples in the hyper cubes around input samples.

        Args:
            inputs (numpy.ndarray): Input samples.
            radius (float): The scope to generate hyper cubes around input samples.

        Returns:
            numpy.ndarray, randomly chosen samples in the hyper cubes.
        """
        LOGGER.debug(TAG, 'enter _generate_hyper_cube().')
        res = []
        for _ in range(self._number_points):
            res.append(np.clip((inputs + np.random.uniform(
                -radius, radius, len(inputs))), 0.0, 1.0).astype(inputs.dtype))
        return np.asarray(res)

    def _rc_forward(self, inputs, radius):
        """
        Generate region-based predictions for input samples.

        Args:
            inputs (numpy.ndarray): Input samples.
            radius (float): The scope to generate hyper cubes around input samples.

        Returns:
            numpy.ndarray, classification result for input samples.
        """
        LOGGER.debug(TAG, 'enter _rc_forward().')
        res = []
        for _, elem in enumerate(inputs):
            hyper_cube_x = self._generate_hyper_cube(elem, radius)
            hyper_cube_preds = []
            for ite_hyper_cube_x in hyper_cube_x:
                model_inputs = Tensor(np.expand_dims(ite_hyper_cube_x, axis=0))
                ite_preds = self._model.predict(model_inputs).asnumpy()[0]
                hyper_cube_preds.append(ite_preds)
            pred_labels = np.argmax(hyper_cube_preds, axis=1)
            bin_count = np.bincount(pred_labels)
            # count the number of different class and choose the max one
            # as final class
            hyper_cube_tag = np.argmax(bin_count, axis=0)
            res.append(hyper_cube_tag)
        return np.asarray(res)

    def detect(self, inputs):
        """
        Tell whether input samples are adversarial or not.

        Args:
            inputs (numpy.ndarray): Suspicious samples to be judged.

        Returns:
            list[int], whether a sample is adversarial. if res[i]=1, then the
            input sample with index i is adversarial.
        """
        LOGGER.debug(TAG, 'enter detect().')
        self._radius = check_param_type('radius', self._radius, float)
        inputs = check_numpy_param('inputs', inputs)
        time_start = time.time()
        res = [1]*inputs.shape[0]
        raw_preds = np.argmax(self._model.predict(Tensor(inputs)).asnumpy(),
                              axis=1)
        rc_preds = self._rc_forward(inputs, self._radius)
        for i in range(inputs.shape[0]):
            if raw_preds[i] == rc_preds[i]:
                res[i] = 0
        LOGGER.debug(TAG,
                     'time used to detect %d samples is : %s seconds',
                     inputs.shape[0],
                     time.time() - time_start)
        return res

    def detect_diff(self, inputs):
        """
        Return raw prediction results and region-based prediction results.

        Args:
            inputs (numpy.ndarray): Input samples.

        Returns:
            numpy.ndarray, raw prediction results and region-based prediction results of input samples.
        """
        LOGGER.debug(TAG, 'enter detect_diff().')
        inputs = check_numpy_param('inputs', inputs)

        raw_preds = self._model.predict(Tensor(inputs))
        rc_preds = self._rc_forward(inputs, self._radius)

        return raw_preds.asnumpy(), rc_preds

    def transform(self, inputs):
        """
        Generate hyper cube for input samples.

        Args:
            inputs (numpy.ndarray): Input samples.

        Returns:
            numpy.ndarray, hyper cube corresponds to every sample.
        """
        LOGGER.debug(TAG, 'enter transform().')
        inputs = check_numpy_param('inputs', inputs)
        res = []
        for _, elem in enumerate(inputs):
            res.append(self._generate_hyper_cube(elem, self._radius))
        return np.asarray(res)
