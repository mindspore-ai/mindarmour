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
Error-Based detector.
"""
import numpy as np
from scipy import stats
from scipy.special import softmax

from mindspore import Tensor
from mindspore import Model

from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_numpy_param, check_model, \
    check_param_in_range, check_param_multi_types, check_int_positive, \
    check_value_positive
from .detector import Detector

LOGGER = LogUtil.get_instance()
TAG = 'MagNet'


class ErrorBasedDetector(Detector):
    """
    The detector reconstructs input samples, measures reconstruction errors and
    rejects samples with large reconstruction errors.

    Reference: `MagNet: a Two-Pronged Defense against Adversarial Examples,
    by Dongyu Meng and Hao Chen, at CCS 2017.
    <https://arxiv.org/abs/1705.09064>`_

    Args:
        auto_encoder (Model): An (trained) auto encoder which
            represents the input by reduced encoding.
        false_positive_rate (float): Detector's false positive rate.
            Default: 0.01.
        bounds (tuple): (clip_min, clip_max). Default: (0.0, 1.0).

    Examples:
        >>> np.random.seed(5)
        >>> ori = np.random.rand(4, 4, 4).astype(np.float32)
        >>> np.random.seed(6)
        >>> adv = np.random.rand(4, 4, 4).astype(np.float32)
        >>> model = Model(Net())
        >>> detector = ErrorBasedDetector(model)
        >>> detector.fit(ori)
        >>> detected_res = detector.detect(adv)
        >>> adv_trans = detector.transform(adv)
    """

    def __init__(self, auto_encoder, false_positive_rate=0.01,
                 bounds=(0.0, 1.0)):
        super(ErrorBasedDetector, self).__init__()
        self._auto_encoder = check_model('auto_encoder', auto_encoder, Model)
        self._false_positive_rate = check_param_in_range('false_positive_rate',
                                                         false_positive_rate,
                                                         0, 1)
        self._threshold = 0.0
        self._bounds = check_param_multi_types('bounds', bounds, [list, tuple])
        for b in self._bounds:
            _ = check_param_multi_types('bound', b, [int, float])

    def fit(self, inputs, labels=None):
        """
        Find a threshold for a given dataset to distinguish adversarial examples.

        Args:
            inputs (numpy.ndarray): Input samples.
            labels (numpy.ndarray): Labels of input samples. Default: None.

        Returns:
            float, threshold to distinguish adversarial samples from benign ones.
        """
        inputs = check_numpy_param('inputs', inputs)

        marks = self.detect_diff(inputs)
        num = int(inputs.shape[0]*self._false_positive_rate)
        marks = np.sort(marks)
        if num <= len(marks):
            self._threshold = marks[-num]
        return self._threshold

    def detect(self, inputs):
        """
        Detect if input samples are adversarial or not.

        Args:
            inputs (numpy.ndarray): Suspicious samples to be judged.

        Returns:
            list[int], whether a sample is adversarial. if res[i]=1, then the
            input sample with index i is adversarial.
        """
        inputs = check_numpy_param('inputs', inputs)
        dist = self.detect_diff(inputs)
        res = [0]*len(dist)
        for i, elem in enumerate(dist):
            if elem > self._threshold:
                res[i] = 1
        return res

    def detect_diff(self, inputs):
        """
        Detect the distance between the original samples and reconstructed samples.

        Args:
            inputs (numpy.ndarray): Input samples.

        Returns:
            float, the distance between reconstructed and original samples.
        """
        inputs = check_numpy_param('inputs', inputs)
        x_trans = self._auto_encoder.predict(Tensor(inputs)).asnumpy()
        diff = np.abs(inputs - x_trans)
        dims = tuple(np.arange(len(inputs.shape))[1:])
        marks = np.mean(np.power(diff, 2), axis=dims)
        return marks

    def transform(self, inputs):
        """
        Reconstruct input samples.

        Args:
            inputs (numpy.ndarray): Input samples.

        Returns:
            numpy.ndarray, reconstructed images.
        """
        inputs = check_numpy_param('inputs', inputs)
        x_trans = self._auto_encoder.predict(Tensor(inputs)).asnumpy()
        if self._bounds is not None:
            clip_min, clip_max = self._bounds
            x_trans = np.clip(x_trans, clip_min, clip_max)
        return x_trans

    def set_threshold(self, threshold):
        """
        Set the parameters threshold.

        Args:
            threshold (float): Detection threshold.
        """
        self._threshold = check_value_positive('threshold', threshold)


class DivergenceBasedDetector(ErrorBasedDetector):
    """
    This class implement a divergence-based detector.

    Reference: `MagNet: a Two-Pronged Defense against Adversarial Examples,
    by Dongyu Meng and Hao Chen, at CCS 2017.
    <https://arxiv.org/abs/1705.09064>`_

    Args:
        auto_encoder (Model): Encoder model.
        model (Model): Targeted model.
        option (str): Method used to calculate Divergence. Default: "jsd".
        t (int): Temperature used to overcome numerical problem. Default: 1.
        bounds (tuple): Upper and lower bounds of data.
            In form of (clip_min, clip_max). Default: (0.0, 1.0).

    Examples:
        >>> np.random.seed(5)
        >>> ori = np.random.rand(4, 4, 4).astype(np.float32)
        >>> np.random.seed(6)
        >>> adv = np.random.rand(4, 4, 4).astype(np.float32)
        >>> encoder = Model(Net())
        >>> model = Model(PredNet())
        >>> detector = DivergenceBasedDetector(encoder, model)
        >>> threshold = detector.fit(ori)
        >>> detector.set_threshold(threshold)
        >>> detected_res = detector.detect(adv)
        >>> adv_trans = detector.transform(adv)
    """

    def __init__(self, auto_encoder, model, option="jsd",
                 t=1, bounds=(0.0, 1.0)):
        super(DivergenceBasedDetector, self).__init__(auto_encoder,
                                                      bounds=bounds)
        self._auto_encoder = auto_encoder
        self._model = check_model('targeted model', model, Model)
        self._threshold = 0.0
        self._option = option
        self._t = check_int_positive('t', t)
        self._bounds = check_param_multi_types('bounds', bounds, [tuple, list])
        for b in self._bounds:
            _ = check_param_multi_types('bound', b, [int, float])

    def detect_diff(self, inputs):
        """
        Detect the distance between original samples and reconstructed samples.

        The distance is calculated by JSD.

        Args:
            inputs (numpy.ndarray): Input samples.

        Returns:
             float, the distance.

        Raises:
            NotImplementedError: If the param `option` is not supported.
        """
        inputs = check_numpy_param('inputs', inputs)
        x_len = inputs.shape[0]
        x_transformed = self._auto_encoder.predict(Tensor(inputs)).asnumpy()
        x_origin = self._model.predict(Tensor(inputs)).asnumpy()
        x_trans = self._model.predict(Tensor(x_transformed)).asnumpy()

        y_pred = softmax(x_origin / self._t, axis=1)
        y_trans_pred = softmax(x_trans / self._t, axis=1)

        if self._option == 'jsd':
            marks = [_jsd(y_pred[i], y_trans_pred[i]) for i in range(x_len)]
        else:
            msg = '{} is not implemented.'.format(self._option)
            LOGGER.error(TAG, msg)
            raise NotImplementedError(msg)
        return np.array(marks)


def _jsd(prob_dist_p, prob_dist_q):
    """
    Compute the Jensen-Shannon Divergence between two probability distributions
        with equal weights.

    Args:
        prob_dist_p (numpy.ndarray): Probability distribution p.
        prob_dist_q (numpy.ndarray): Probability distribution q.

    Returns:
        float, the Jensen-Shannon Divergence.
    """
    prob_dist_p = check_numpy_param('prob_dist_p', prob_dist_p)
    prob_dist_q = check_numpy_param('prob_dist_q', prob_dist_q)
    norm_dist_p = prob_dist_p / (np.linalg.norm(prob_dist_p, ord=1) + 1e-12)
    norm_dist_q = prob_dist_q / (np.linalg.norm(prob_dist_q, ord=1) + 1e-12)
    norm_mean = 0.5*(norm_dist_p + norm_dist_q)
    return 0.5*(stats.entropy(norm_dist_p, norm_mean)
                + stats.entropy(norm_dist_q, norm_mean))
