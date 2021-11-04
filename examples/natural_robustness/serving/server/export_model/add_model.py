# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================
"""add model generator"""
import os
from shutil import copyfile
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    """Define Net of add"""

    def __init__(self):
        super(Net, self).__init__()
        self.add = ops.Add()

    def construct(self, x_, y_):
        """construct add net"""
        return self.add(x_, y_)


def export_net():
    """Export add net of 2x2 + 2x2, and copy output model `tensor_add.mindir` to directory ../add/1"""
    x = np.ones([2, 2]).astype(np.float32)
    y = np.ones([2, 2]).astype(np.float32)
    add = Net()
    ms.export(add, ms.Tensor(x), ms.Tensor(y), file_name='tensor_add', file_format='MINDIR')
    dst_dir = '../perturbation/1'
    try:
        os.mkdir(dst_dir)
    except OSError:
        pass

    dst_file = os.path.join(dst_dir, 'tensor_add.mindir')
    copyfile('tensor_add.mindir', dst_file)
    print("copy tensor_add.mindir to " + dst_dir + " success")


if __name__ == "__main__":
    export_net()
