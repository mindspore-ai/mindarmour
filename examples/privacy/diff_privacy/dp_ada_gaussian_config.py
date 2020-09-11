# Copyright 2020 Huawei Technologies Co., Ltd
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
"""
network config setting, will be used in train.py
"""

from easydict import EasyDict as edict

mnist_cfg = edict({
    'num_classes': 10,  # the number of classes of model's output
    'lr': 0.01,  # the learning rate of model's optimizer
    'momentum': 0.9,  # the momentum value of model's optimizer
    'epoch_size': 5,  # training epochs
    'batch_size': 256,  # batch size for training
    'image_height': 32,  # the height of training samples
    'image_width': 32,  # the width of training samples
    'save_checkpoint_steps': 234,  # the interval steps for saving checkpoint file of the model
    'keep_checkpoint_max': 10,  # the maximum number of checkpoint files would be saved
    'device_target': 'Ascend',  # device used
    'data_path': '../../common/dataset/MNIST',  # the path of training and testing data set
    'dataset_sink_mode': False,  # whether deliver all training data to device one time
    'micro_batches': 32,  # the number of small batches split from an original batch
    'norm_bound': 1.0,  # the clip bound of the gradients of model's training parameters
    'initial_noise_multiplier': 0.05,  # the initial multiplication coefficient of the noise added to training
    # parameters' gradients
    'noise_mechanisms': 'AdaGaussian',  # the method of adding noise in gradients while training
    'optimizer': 'Momentum'  # the base optimizer used for Differential privacy training
})
