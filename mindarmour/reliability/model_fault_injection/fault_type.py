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
"""
Fault type module
"""

import math
import random
from struct import pack, unpack
import numpy as np

from mindarmour.utils.logger import LogUtil

LOGGER = LogUtil.get_instance()
TAG = 'FaultType'


class FaultType:
    """Implementation of specified fault type."""
    @staticmethod
    def _bitflip(value, pos):
        """
        Implement of bitflip.
        Args:
            value (numpy.ndarray): Input data.
            pos (list): The index of flip position.

        Returns:
            numpy.ndarray, bitflip data.
        """
        bits = str(value.dtype)[-2:] if str(value.dtype)[-2].isdigit() else str(value.dtype)[-1]
        value_format = 'B' * int(int(bits) / 8)
        value_bytes = value.tobytes()
        bytes_ = list(unpack(value_format, value_bytes))
        for p in pos:
            [q, r] = divmod(p, 8)
            bytes_[q] ^= 1 << r
        new_value_bytes = pack(value_format, *bytes_)
        new_value = np.frombuffer(new_value_bytes, value.dtype)
        return new_value[0]

    def _fault_inject(self, value, fi_type, fi_size):
        """
        Inject the specified fault into the randomly chosen values.
        For zeros, anti_activation and precision_loss, fi_size is the percentage of
        total number. And the others fault, fi_size is the exact number of values to
        be injected.
        Args:
            value (numpy.ndarray): Input data.
            fi_type (str): Fault type.
            fi_size (int): The number of fault injection.

        Returns:
            numpy.ndarray, data after fault injection.
        """
        num = value.size
        if fi_type in ['zeros', 'anti_activation', 'precision_loss']:
            change_size = (fi_size * num) / 100
            change_size = math.floor(change_size)
        else:
            change_size = fi_size

        if change_size > num:
            change_size = num
        # Choose the indices for FI
        ind = random.sample(range(num), change_size)

        # got specified fault type
        try:
            func = getattr(self, fi_type)
            value = func(value, ind)
            return value
        except AttributeError:
            msg = "'Undefined fault type', got {}.".format(fi_type)
            LOGGER.error(TAG, msg)
            raise AttributeError(msg)

    def _bitflips_random(self, value, fi_indices):
        """
        Flip bit randomly for specified value.
        Args:
            value (numpy.ndarray): Input data.
            fi_indices (list): The index of injected data.

        Returns:
            numpy.ndarray, data after fault injection.
        """
        for item in fi_indices:
            val = value[item]
            pos = random.sample(range(int(str(val.dtype)[-2:])),
                                1 if np.random.random() < 0.618 else 2)
            val_new = self._bitflip(val, pos)
            value[item] = val_new
        return value

    def _bitflips_designated(self, value, fi_indices):
        """
        Flip the key bit for specified value.

        Args:
            value (numpy.ndarray): Input data.
            fi_indices (list): The index of injected data.

        Returns:
            numpy.ndarray, data after fault injection.
        """
        for item in fi_indices:
            val = value[item]
            # uint8 uint16 uint32 uint64
            bits = str(value.dtype)[-2:] if str(value.dtype)[-2].isdigit() else str(value.dtype)[-1]
            if 'uint' in str(val.dtype):
                pos = int(bits) - 1
            # int8 int16 int32 int64 float16 float32 float64
            else:
                pos = int(bits) - 2
            val_new = self._bitflip(val, [pos])
            value[item] = val_new
        return value

    @staticmethod
    def _random(value, fi_indices):
        """
        Reset specified value randomly, range from -1 to 1.
        Args:
            value (numpy.ndarray): Input data.
            fi_indices (list): The index of injected data.

        Returns:
            numpy.ndarray, data after fault injection.
        """
        for item in fi_indices:
            value[item] = np.random.random() * 2 - 1
        return value

    @staticmethod
    def _zeros(value, fi_indices):
        """
        Set specified value into zeros.
        Args:
            value (numpy.ndarray): Input data.
            fi_indices (list): The index of injected data.

        Returns:
            numpy.ndarray, data after fault injection.
        """
        value[fi_indices] = 0.
        return value

    @staticmethod
    def _nan(value, fi_indices):
        """
        Set specified value into nan.
        Args:
            value (numpy.ndarray): Input data.
            fi_indices (list): The index of injected data.

        Returns:
            numpy.ndarray, data after fault injection.
        """
        try:
            value[fi_indices] = np.nan
            return value
        except ValueError:
            return value

    @staticmethod
    def _inf(value, fi_indices):
        """
        Set specified value into inf
        Args:
            value (numpy.ndarray): Input data.
            fi_indices (list): The index of injected data.

        Returns:
            numpy.ndarray, data after fault injection.
        """
        try:
            value[fi_indices] = np.inf
            return value
        except OverflowError:
            return value

    @staticmethod
    def _anti_activation(value, fi_indices):
        """
        Minus specified value.
        Args:
            value (numpy.ndarray): Input data.
            fi_indices (list): The index of injected data.

        Returns:
            numpy.ndarray, data after fault injection.
        """
        value[fi_indices] = -value[fi_indices]
        return value

    @staticmethod
    def _precision_loss(value, fi_indices):
        """
        Round specified value, round to 1 decimal place.
        Args:
            value (numpy.ndarray): Input data.
            fi_indices (list): The index of injected data.

        Returns:
            numpy.ndarray, data after fault injection.
        """
        value[fi_indices] = np.around(value[fi_indices], decimals=1)
        return value
