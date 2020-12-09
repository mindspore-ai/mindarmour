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
"""
python lenet5_dp_pynative_model.py --data_path /YourDataPath --micro_batches=2
"""
import os

import mindspore.nn as nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import LossMonitor
from mindspore.nn.metrics import Accuracy
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
import mindspore.common.dtype as mstype

from mindarmour.privacy.diff_privacy import DPModel
from mindarmour.privacy.diff_privacy import PrivacyMonitorFactory
from mindarmour.privacy.diff_privacy import DPOptimizerClassFactory
from mindarmour.privacy.diff_privacy import ClipMechanismsFactory
from mindarmour.utils.logger import LogUtil
from examples.common.networks.lenet5.lenet5_net import LeNet5
from lenet5_config import mnist_cfg as cfg

LOGGER = LogUtil.get_instance()
LOGGER.set_level('INFO')
TAG = 'Lenet5_train'


def generate_mnist_dataset(data_path, batch_size=32, repeat_size=1,
                           num_parallel_workers=1, sparse=True):
    """
    create dataset for training or testing
    """
    # define dataset
    ds1 = ds.MnistDataset(data_path)

    # define operation parameters
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width),
                          interpolation=Inter.LINEAR)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # apply map operations on images
    if not sparse:
        one_hot_enco = C.OneHot(10)
        ds1 = ds1.map(input_columns="label", operations=one_hot_enco,
                      num_parallel_workers=num_parallel_workers)
        type_cast_op = C.TypeCast(mstype.float32)
    ds1 = ds1.map(input_columns="label", operations=type_cast_op,
                  num_parallel_workers=num_parallel_workers)
    ds1 = ds1.map(input_columns="image", operations=resize_op,
                  num_parallel_workers=num_parallel_workers)
    ds1 = ds1.map(input_columns="image", operations=rescale_op,
                  num_parallel_workers=num_parallel_workers)
    ds1 = ds1.map(input_columns="image", operations=hwc2chw_op,
                  num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    ds1 = ds1.shuffle(buffer_size=buffer_size)
    ds1 = ds1.batch(batch_size, drop_remainder=True)
    ds1 = ds1.repeat(repeat_size)

    return ds1


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)
    network = LeNet5()
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet",
                                 directory='./trained_ckpt_file/',
                                 config=config_ck)

    # get training dataset
    ds_train = generate_mnist_dataset(os.path.join(cfg.data_path, "train"),
                                      cfg.batch_size)

    if cfg.micro_batches and cfg.batch_size % cfg.micro_batches != 0:
        raise ValueError("Number of micro_batches should divide evenly batch_size")
    # Create a factory class of DP mechanisms, this method is adding noise in gradients while training.
    # Mechanisms can be 'Gaussian' or 'AdaGaussian', in which noise
    # would be decayed with 'AdaGaussian' mechanism while be constant with 'Gaussian' mechanism.
    dp_opt = DPOptimizerClassFactory(micro_batches=cfg.micro_batches)
    dp_opt.set_mechanisms(cfg.noise_mechanisms,
                          norm_bound=cfg.norm_bound,
                          initial_noise_multiplier=cfg.initial_noise_multiplier,
                          decay_policy=None)
    # Create a factory class of clip mechanisms, this method is to adaptive clip
    # gradients while training, decay_policy support 'Linear' and 'Geometric',
    # learning_rate is the learning rate to update clip_norm,
    # target_unclipped_quantile is the target quantile of norm clip,
    # fraction_stddev is the stddev of Gaussian normal which used in
    # empirical_fraction, the formula is
    # $empirical_fraction + N(0, fraction_stddev)$.
    clip_mech = ClipMechanismsFactory().create(cfg.clip_mechanisms,
                                               decay_policy=cfg.clip_decay_policy,
                                               learning_rate=cfg.clip_learning_rate,
                                               target_unclipped_quantile=cfg.target_unclipped_quantile,
                                               fraction_stddev=cfg.fraction_stddev)
    net_opt = dp_opt.create('Momentum')(params=network.trainable_params(), learning_rate=cfg.lr, momentum=cfg.momentum)
    # Create a monitor for DP training. The function of the monitor is to compute and print the privacy budget(eps
    # and delta) while training.
    rdp_monitor = PrivacyMonitorFactory.create('rdp',
                                               num_samples=60000,
                                               batch_size=cfg.batch_size,
                                               initial_noise_multiplier=cfg.initial_noise_multiplier*cfg.norm_bound,
                                               per_print_times=10)
    # Create the DP model for training.
    model = DPModel(micro_batches=cfg.micro_batches,
                    norm_bound=cfg.norm_bound,
                    noise_mech=None,
                    clip_mech=clip_mech,
                    network=network,
                    loss_fn=net_loss,
                    optimizer=net_opt,
                    metrics={"Accuracy": Accuracy()})

    LOGGER.info(TAG, "============== Starting Training ==============")
    model.train(cfg['epoch_size'], ds_train, callbacks=[ckpoint_cb, LossMonitor(), rdp_monitor],
                dataset_sink_mode=cfg.dataset_sink_mode)

    LOGGER.info(TAG, "============== Starting Testing ==============")
    ckpt_file_name = 'trained_ckpt_file/checkpoint_lenet-5_234.ckpt'
    param_dict = load_checkpoint(ckpt_file_name)
    load_param_into_net(network, param_dict)
    ds_eval = generate_mnist_dataset(os.path.join(cfg.data_path, 'test'), batch_size=cfg.batch_size)
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    LOGGER.info(TAG, "============== Accuracy: %s  ==============", acc)
