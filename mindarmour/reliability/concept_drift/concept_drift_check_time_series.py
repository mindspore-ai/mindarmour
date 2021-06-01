# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Concpt drift module
"""

from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt
from mindarmour.utils._check_param import _check_array_not_empty
from mindarmour.utils._check_param import check_param_type, check_param_in_range


class ConceptDriftCheckTimeSeries:
    """
    ConceptDriftCheckTimeSeries is used for example series distribution change detection.

    Args:
        window_size(int): Size of a concept window, no less than 10. If given the input data,
            window_size belongs to [10, 1/3*len(input data)]. If the data is periodic, usually
            window_size equals 2-5 periods, such as, for monthly/weekly data, the data volume
            of 30/7 days is a period. Default: 100.
        rolling_window(int): Smoothing window size, belongs to [1, window_size]. Default:10.
        step(int): The jump length of the sliding window, belongs to [1, window_size]. Default:10.
        threshold_index(float): The threshold index, :math:`(-\infty, +\infty)`. Default: 1.5.
        need_label(bool): False or True. If need_label=True, concept drift labels are needed.
            Default: False.

    Examples:
        >>> concept = ConceptDriftCheckTimeSeries(window_size=100, rolling_window=10,
        >>>                   step=10, threshold_index=1.5, need_label=False)
        >>> data_example = 5*np.random.rand(1000)
        >>> data_example[200: 800] = 20*np.random.rand(600)
        >>> score, threshold, concept_drift_location = concept.concept_check(data_example)
    """

    def __init__(self, window_size=100, rolling_window=10,
                 step=10, threshold_index=1.5, need_label=False):
        self.window_size = check_param_type('window_size', window_size, int)
        self.rolling_window = check_param_type('rolling_window', rolling_window, int)
        self.step = check_param_type('step', step, int)
        self.threshold_index = check_param_type('threshold_index', threshold_index, float)
        self.need_label = check_param_type('need_label', need_label, bool)
        self._in_size = window_size
        self._out_size = window_size
        self._res_size = int(0.1*window_size)

    def _reservoir_model_feature(self, window_data):
        """
        Extract example features in reservoir model.

        Args:
            window_data(numpy.ndarray): The input data (in one window).

        Returns:
            - numpy.ndarray, the output weight of reservoir model.
            - numpy.ndarray, the state of the reservoir model in the latent space.

        Examples:
            >>> input_data = np.random.rand(100)
            >>> w, x = ConceptDriftCheckTimeSeries._reservoir_model_feature(window_data)
        """
        # Initialize weights
        res_size = self._res_size
        x_state = _w_generate(res_size, len(window_data), window_data)
        x_state_t = x_state.T
        # Data reshape
        data_channel = None
        if window_data.ndim == 2:
            data_channel = window_data.shape[1]
        if window_data.ndim == 1:
            data_channel = 1
        y_t = window_data.reshape(len(window_data), data_channel)
        reg = 1e-8
        # Calculate w_out
        w_out = np.dot(np.dot(y_t, x_state_t),
                       np.linalg.inv(np.dot(x_state, x_state_t) + reg*np.eye(res_size)))
        return w_out, x_state

    def _concept_distance(self, data_x, data_y):
        """
        Calculate the distance of two example blocks.

        Args:
            data_x(numpy.ndarray): Data x.
            data_y(numpy.ndarray): Data y.

        Returns:
            - float, distance between data_x and data_y.

        Examples:
            >>> x = np.random.rand(100)
            >>> y = np.random.rand(100)
            >>> score = ConceptDriftCheckTimeSeries._concept_distance(x, y)
        """
        # Feature extraction
        feature_x = self._reservoir_model_feature(data_x)
        feature_y = self._reservoir_model_feature(data_y)
        # Calculate distance
        distance_wx = sum(abs(np.dot(feature_x[0], feature_x[1])
                              - np.dot(feature_y[0], feature_y[1])))/len(data_x)
        statistic_feature = abs(data_x.mean() - data_y.mean()).mean()
        distance_score = (distance_wx + statistic_feature) / (1 + distance_wx + statistic_feature)
        distance_score_mean = distance_score.mean()
        return distance_score_mean

    def _data_process(self, data):
        """
        Data processing.

        Args:
            data(numpy.ndarray): Input data.

        Returns:
            - numpy.ndarray, data after smoothing.

        Examples:
            >>> data_example = np.random.rand(100)
            >>> data_example = ConceptDriftCheckTimeSeries._data_process(data_example)
        """
        temp = []
        data_channel = None
        if data.ndim == 2:
            data_channel = data.shape[1]
        if data.ndim == 1:
            data_channel = 1
        data = data.reshape(len(data), data_channel)
        # Moving average
        for i in range(data_channel):
            data_av = np.convolve(data[:, i],
                                  np.ones((self.rolling_window,)) / self.rolling_window, 'valid')
            data_av = np.append(data_av, np.ones(self.rolling_window - 1)*data_av[-1])
            data_av = (data_av - data_av.min()) / (data_av.max() - data_av.min())
            temp.append(data_av)
        smooth_data = np.array(temp).T
        return smooth_data

    def concept_check(self, data):
        """
        Find concept drift locations in a data series.

        Args:
            data(numpy.ndarray): Input data. The shape of data could be (n,1) or (n,m).
                Note that each column (m columns) is one data series.

        Returns:
            - numpy.ndarray, the concept drift score of the example series.
            - float, the threshold to judge concept drift.
            - list, the location of the concept drift.

        Examples：
            >>> concept = ConceptDriftCheckTimeSeries(window_size=100, rolling_window=10,
            >>>                   step=10, threshold_index=1.5, need_label=False)
            >>> data_example = 5*np.random.rand(1000)
            >>> data_example[200: 800] = 20*np.random.rand(600)
            >>> score, drift_threshold, drift_location = concept.concept_check(data_example)
        """
        # data check
        data = _check_array_not_empty('data', data)
        data = check_param_type('data', data, np.ndarray)
        check_param_in_range('window_size', self.window_size, 10, int((1 / 3)*len(data)))
        check_param_in_range('rolling_window', self.rolling_window, 1, self.window_size)
        check_param_in_range('step', self.step, 1, self.window_size)
        original_data = data
        data = self._data_process(data)
        # calculate drift score
        drift_score = np.zeros(len(data))
        step_size = self.step
        for i in range(0, len(data) - 2*self.window_size, step_size):
            data_x = data[i: i + self.window_size]
            data_y = data[i + self.window_size:i + 2*self.window_size]
            drift_score[i + self.window_size] = self._concept_distance(data_x, data_y)
        threshold = _cal_threshold(drift_score, self.threshold_index)
        # original label
        label, label_location = _original_label(data, threshold, drift_score,
                                                self.window_size, step_size)
        # label continue
        label_continue = _label_continue_process(label)
        # find drift blocks
        concept_drift_location, drift_point = _drift_blocks(drift_score,
                                                            label_continue, label_location)
        # save result
        _result_save(original_data, threshold, concept_drift_location, drift_point, drift_score)
        return drift_score, threshold, concept_drift_location


def _result_save(original_data, threshold, concept_location, drift_point, drift_score):
    """
    To save the result.

    Args:
        original_data(numpy.ndarray): The input data.
        threshold(float): The concept drift threshold.
        concept_location(list): The concept drift locations(x-axis).
        drift_point(list): The precise drift point of a drift.
        drift_score(numpy.ndarray): The drift score of input data.
    """
    plt.figure(figsize=(20, 8))
    plt.subplot(2, 1, 1)
    # Plot input data and drift points
    plt.plot(original_data, label="data")
    plt.title('concept drift check, threshold=' + str(threshold), fontsize=25)
    plt.scatter(concept_location, np.ones(len(concept_location)),
                marker='*', s=200, color="b", label="concept drift occurred")
    for _, i in enumerate(drift_point):
        plt.axvline(x=i, color='r', linestyle='--')
    plt.legend(fontsize=15)
    plt.subplot(2, 1, 2)
    # Plot drift score
    plt.plot(drift_score, label="drift_score")
    plt.axhline(y=threshold, color='r', linestyle='--', label="threshold")
    plt.legend(fontsize=15)
    plt.savefig('concept_drift_check.pdf')


def _original_label(original_data, threshold, drift_score, window_size, step_size):
    """
    To obtain a original drift label of time series.

    Args:
        original_data(numpy.ndarray): The input data.
        threshold(float): The drift threshold.
        drift_score(numpy.ndarray): The drift score of the input data.
        window_size(int): Size of a concept window.
            Usually 3 periods of the input data if it is periodic.
        step_size(int): The jump length of the sliding window.

    Returns:
        - list, the drift label of input data.
            0 means no drift, and 1 means drift happens.
        - list, the locations of drifts(x-axis).
    """
    label = []
    label_location = []
    # Label: label=0, no drifts; label=1, drift happens.
    for i in range(0, len(original_data) - 2*window_size, step_size):
        label_location.append(i + window_size)
        if drift_score[i + window_size] >= threshold:
            label.append(1)
        else:
            label.append(0)
    return label, label_location


def _label_continue_process(label):
    """
    To obtain a continual drift label of time series.

    Args:
        label(list): The original drift label.

    Returns:
        - numpy.ndarray, The continual drift label.
            The drift may happen occasionally, we hope to avoid
            frequent alarms, so label continue process is necessary.
    """
    if label[-1] == 1 and label[-2] == 0 and label[-3] == 0 and label[-4] == 0:
        label[-1] = 0
    if label[0] == 1 and label[1] == 0 and label[2] == 0 and label[3] == 0:
        label[0] = 0
    label_continue = np.array(label)
    # Label continue process
    for i in range(1, len(label) - 1):
        if label[i - 1] == 0 and label[i + 1] == 0:
            label_continue[i - 1:i + 1] = 0
    return label_continue


def _find_loc(label_location):
    return label_location[1] - label_location[0]


def _continue_block(location):
    """
    Find continue blocks of concept drift.

    Args:
        location(numpy.ndarray): The locations of concept drift.

    Returns:
        - list, continue blocks of concept drift.
    """
    area = []
    for _, loc in groupby(enumerate(location), _find_loc):
        l_1 = [j for i, j in loc]
        area.append(l_1)
    return area


def _drift_blocks(drift_score, label_continue, label_location):
    """
    Find the concept drift areas.

    Args:
        drift_score(numpy.ndarray): The drift score of the data series.
        label_continue(numpy.ndarray):  The concept drift continual label.
        label_location(numpy.ndarray):  The locations of concept drift(x-axis).

    Returns:
        - list, the concept drift locations(x-axis) after continual blocks finding.
        - list, return a precise beginning location of a drift.
    """
    # Find drift blocks
    area = _continue_block(np.where(label_continue == 1)[0])
    label_continue = np.array(label_continue)
    label_location = np.array(label_location)
    label_continue = label_continue[label_continue == 1]
    concept_location = []
    drift_point = []
    # Find drift points
    for _, lo_ in enumerate(area):
        location = label_location[lo_]
        concept_location.extend(location)
        if label_continue.size > 0:
            drift_point.append(location[np.where(drift_score[location]
                                                 == np.max(drift_score[location]))[0]])
        else:
            drift_point.append(None)
    return concept_location, drift_point


def _w_generate(res_size, in_size, input_data):
    """
    Randomly generate weights of the reservoir model.

    Args:
        res_size(int): The number of reservoir nodes.
        in_size(int): The input size of reservoir model.
        input_data(numpy.ndarray): Input data.

    Returns:
        - numpy.ndarray, the state of reservoir.
    """
    # Weight generates randomly
    np.random.seed(42)
    w_in = np.random.rand(res_size, in_size) - 0.5
    w_0 = np.random.rand(res_size, res_size) - 0.5
    w_0 *= 0.8
    a_speed = 0.3
    # Data reshape
    data_channel = None
    if input_data.ndim == 2:
        data_channel = input_data.shape[1]
    if input_data.ndim == 1:
        data_channel = 1
    # Reservoir state
    x_state = np.zeros((res_size, data_channel))
    u_input = input_data.reshape(len(input_data), data_channel)
    for _ in range(50):
        x_state = (1 - a_speed)*x_state + \
                  a_speed*np.tanh(np.dot(w_in, u_input) + np.dot(w_0, x_state))
    return x_state


def _cal_distance(matrix1, matrix2):
    """
    Calculate distance between two matrix.

    Args:
        matrix1(numpy.ndarray): Input array.
        matrix2(numpy.ndarray): Input array.

    Returns:
        - numpy.ndarray, distance between two arrays.
    """
    w_mean_x = np.mean(matrix1, axis=0)
    w_mean_y = np.mean(matrix2, axis=0)
    distance = sum(abs(w_mean_x - w_mean_y))
    return distance


def _cal_threshold(distance, threshold_index):
    """
    Calculate the threshold of concept drift.

    Args:
        distance(numpy.ndarray): The distance between two data series.
        threshold_index(float): Threshold adjusted index, [-∞, +∞].

    Returns:
        - float, [0, 1].
    """
    distance = distance[distance > 0]
    # Threshold calculation
    if distance.size > 0:
        q_1 = np.percentile(distance, 25)
        q_3 = np.percentile(distance, 75)
        q_diff = q_3 - q_1
        threshold = np.clip(0.1 + threshold_index*q_diff, 0, 1)
    else:
        threshold = 1
    return threshold
