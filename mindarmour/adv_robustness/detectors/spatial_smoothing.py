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
Spatial-Smoothing detector.
"""
import numpy as np
from scipy import ndimage

from mindspore import Model
from mindspore import Tensor

from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_model, check_numpy_param, \
    check_pair_numpy_param, check_int_positive, check_param_type, \
    check_param_in_range, check_equal_shape, check_value_positive
from .detector import Detector

LOGGER = LogUtil.get_instance()
TAG = 'SpatialSmoothing'


def _median_filter_np(inputs, size=2):
    """median filter using numpy"""
    return ndimage.filters.median_filter(inputs, size=size, mode='reflect')


class SpatialSmoothing(Detector):
    """
    Detect method based on spatial smoothing.

    Args:
        model (Model): Target model.
        ksize (int): Smooth window size. Default: 3.
        is_local_smooth (bool): If True, trigger local smooth. If False, none
            local smooth. Default: True.
        metric (str): Distance method. Default: 'l1'.
        false_positive_ratio (float): False positive rate over
            benign samples. Default: 0.05.

    Examples:
        >>> detector = SpatialSmoothing(model)
        >>> detector.fit(ori, labels)
        >>> adv_ids = detector.detect(adv)
    """

    def __init__(self, model, ksize=3, is_local_smooth=True,
                 metric='l1', false_positive_ratio=0.05):
        super(SpatialSmoothing, self).__init__()
        self._ksize = check_int_positive('ksize', ksize)
        self._is_local_smooth = check_param_type('is_local_smooth',
                                                 is_local_smooth,
                                                 bool)
        self._model = check_model('model', model, Model)
        self._metric = metric
        self._fpr = check_param_in_range('false_positive_ratio',
                                         false_positive_ratio,
                                         0, 1)
        self._threshold = None

    def fit(self, inputs, labels=None):
        """
        Train detector to decide the threshold. The proper threshold make
        sure the actual false positive rate over benign sample is less than
        the given value.

        Args:
            inputs (numpy.ndarray): Benign samples.
            labels (numpy.ndarray): Default None.

        Returns:
            float, threshold, distance larger than which is reported
            as positive, i.e. adversarial.
        """
        inputs = check_numpy_param('inputs', inputs)
        raw_pred = self._model.predict(Tensor(inputs)).asnumpy()
        smoothing_pred = self._model.predict(Tensor(self.transform(inputs))).asnumpy()

        dist = self._dist(raw_pred, smoothing_pred)
        index = int(len(dist)*(1 - self._fpr))
        threshold = np.sort(dist, axis=None)[index]
        self._threshold = threshold
        return self._threshold

    def detect(self, inputs):
        """
        Detect if an input sample is an adversarial example.

        Args:
            inputs (numpy.ndarray): Suspicious samples to be judged.

        Returns:
            list[int], whether a sample is adversarial. if res[i]=1, then the
            input sample with index i is adversarial.
        """
        inputs = check_numpy_param('inputs', inputs)
        raw_pred = self._model.predict(Tensor(inputs)).asnumpy()
        smoothing_pred = self._model.predict(Tensor(self.transform(inputs))).asnumpy()
        dist = self._dist(raw_pred, smoothing_pred)

        res = [0]*len(dist)
        for i, elem in enumerate(dist):
            if elem > self._threshold:
                res[i] = 1

        return res

    def detect_diff(self, inputs):
        """
        Return the raw distance value (before apply the threshold) between
        the input sample and its smoothed counterpart.

        Args:
            inputs (numpy.ndarray): Suspicious samples to be judged.

        Returns:
            float, distance.
        """
        inputs = check_numpy_param('inputs', inputs)
        raw_pred = self._model.predict(Tensor(inputs)).asnumpy()
        smoothing_pred = self._model.predict(Tensor(self.transform(inputs))).asnumpy()
        dist = self._dist(raw_pred, smoothing_pred)
        return dist

    def transform(self, inputs):
        inputs = check_numpy_param('inputs', inputs)
        return _median_filter_np(inputs, self._ksize)

    def set_threshold(self, threshold):
        """
        Set the parameters threshold.

        Args:
            threshold (float): Detection threshold.
        """
        self._threshold = check_value_positive('threshold', threshold)

    def _dist(self, before, after):
        """
        Calculate the distance between the model outputs of a raw sample and
            its smoothed counterpart.

        Args:
            before (numpy.ndarray): Model output of raw samples.
            after (numpy.ndarray): Model output of smoothed counterparts.

        Returns:
            float, distance based on specified norm.
        """
        before, after = check_pair_numpy_param('before', before, 'after', after)
        before, after = check_equal_shape('before', before, 'after', after)
        res = []
        diff = after - before
        for _, elem in enumerate(diff):
            if self._metric == 'l1':
                res.append(np.linalg.norm(elem, ord=1))
            elif self._metric == 'l2':
                res.append(np.linalg.norm(elem, ord=2))
            else:
                res.append(np.linalg.norm(elem, ord=1))
        return res
