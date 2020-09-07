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
Radar map.
"""
from math import pi

import numpy as np

import matplotlib.pyplot as plt

from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_param_type, check_numpy_param, \
    check_param_multi_types, check_equal_length

LOGGER = LogUtil.get_instance()
TAG = 'RadarMetric'


class RadarMetric:
    """
    Radar chart to show the robustness of a model by multiple metrics.

    Args:
        metrics_name (Union[tuple, list]): An array of names of metrics to show.
        metrics_data (numpy.ndarray): The (normalized) values of each metrics of
            multiple radar curves, like [[0.5, 0.8, ...], [0.2,0.6,...], ...].
            Each set of values corresponds to one radar curve.
        labels (Union[tuple, list]): Legends of all radar curves.
        title (str): Title of the chart.
        scale (str): Scalar to adjust axis ticks, such as 'hide', 'norm',
            'sparse' or 'dense'. Default: 'hide'.

    Raises:
        ValueError: If scale not in ['hide', 'norm', 'sparse', 'dense'].

    Examples:
        >>> metrics_name = ['MR', 'ACAC', 'ASS', 'NTE', 'ACTC']
        >>> def_metrics = [0.9, 0.85, 0.6, 0.7, 0.8]
        >>> raw_metrics = [0.5, 0.3, 0.55, 0.65, 0.7]
        >>> metrics_data = [def_metrics, raw_metrics]
        >>> metrics_labels = ['before', 'after']
        >>> rm = RadarMetric(metrics_name,
        >>>                  metrics_data,
        >>>                  metrics_labels,
        >>>                  title='',
        >>>                  scale='sparse')
        >>> rm.show()
    """

    def __init__(self, metrics_name, metrics_data, labels, title, scale='hide'):

        self._metrics_name = check_param_multi_types('metrics_name',
                                                     metrics_name,
                                                     [tuple, list])
        self._metrics_data = check_numpy_param('metrics_data', metrics_data)
        self._labels = check_param_multi_types('labels', labels, (tuple, list))

        _, _ = check_equal_length('metrics_name', metrics_name,
                                  'metrics_data', self._metrics_data[0])
        _, _ = check_equal_length('labels', labels, 'metrics_data', metrics_data)
        self._title = check_param_type('title', title, str)
        if scale in ['hide', 'norm', 'sparse', 'dense']:
            self._scale = scale
        else:
            msg = "scale must be in ['hide', 'norm', 'sparse', 'dense'], but " \
                  "got {}".format(scale)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

        self._nb_var = len(metrics_name)
        # divide the plot / number of variable
        self._angles = [n / self._nb_var*2.0*pi for n in
                        range(self._nb_var)]
        self._angles += self._angles[:1]

        # add one more point
        data = [self._metrics_data, self._metrics_data[:, [0]]]
        self._metrics_data = np.concatenate(data, axis=1)

    def show(self):
        """
        Show the radar chart.
        """
        # Initialise the spider plot
        plt.clf()
        axis_pic = plt.subplot(111, polar=True)
        axis_pic.spines['polar'].set_visible(False)
        axis_pic.set_yticklabels([])

        # If you want the first axis to be on top:
        axis_pic.set_theta_offset(pi / 2)
        axis_pic.set_theta_direction(-1)

        # Draw one axe per variable + add labels labels yet
        plt.xticks(self._angles[:-1], self._metrics_name)

        # Draw y labels
        axis_pic.set_rlabel_position(0)
        if self._scale == 'hide':
            plt.yticks([0.0], color="grey", size=7)
        elif self._scale == 'norm':
            plt.yticks([0.2, 0.4, 0.6, 0.8],
                       ["0.2", "0.4", "0.6", "0.8"],
                       color="grey", size=7)
        elif self._scale == 'sparse':
            plt.yticks([0.5], ["0.5"], color="grey", size=7)
        elif self._scale == 'dense':
            ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            labels = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6",
                      "0.7", "0.8", "0.9"]
            plt.yticks(ticks, labels, color="grey", size=7)
        else:
            # default
            plt.yticks([0.0], color="grey", size=7)
        plt.ylim(0, 1)

        # plot border
        axis_pic.plot(self._angles, [1]*(self._nb_var + 1), color='grey',
                      linewidth=1, linestyle='solid')

        for i in range(len(self._labels)):
            axis_pic.plot(self._angles, self._metrics_data[i], linewidth=1,
                          linestyle='solid', label=self._labels[i])
            axis_pic.fill(self._angles, self._metrics_data[i], alpha=0.1)

        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0., 0.))
        plt.title(self._title, y=1.1, color='g')
        plt.show()
