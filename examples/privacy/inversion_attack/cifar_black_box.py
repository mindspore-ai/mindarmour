# Copyright 2024 Huawei Technologies Co., Ltd
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
Examples of model inversion attack
"""
import numpy as np
import matplotlib.pyplot as plt
import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Tensor, context
from mindarmour.privacy.evaluation.model_inversion_attack import ModelInversionAttack
from mindarmour.utils.logger import LogUtil
from examples.privacy.inversion_attack.inversion_net import CIFAR10CNNDecoderConv11
from examples.common.networks.cifar10cnn.cifar10cnn_net import CIFAR10CNN
from examples.common.dataset.data_processing import generate_dataset_cifar
LOGGER = LogUtil.get_instance()
LOGGER.set_level('INFO')
TAG = 'ModelInversionAttack'


# pylint: disable=invalid-name


def cifar_inversion_attack(net, inv_net, ckptpath):
    """
    Image inversion attack based on CNN and CAFIR10 dataset.
    Args:
        net(Cell): target model
        inv_net(Cell): atk model for model inversion
        ckptpath(str): ckpt file for target model
    """
    # upload trained network

    load_dict = load_checkpoint(ckptpath)
    load_param_into_net(net, load_dict)

    # get original data and their inferred fearures
    data_list = "../../common/dataset/CIFAR10" #/train
    ds = generate_dataset_cifar(data_list, 32, usage="train", repeat_num=1)
    data_list = "../../common/dataset/CIFAR10" #/test
    ds_test = generate_dataset_cifar(data_list, 32, usage="test", repeat_num=1)
    i = 0
    batch_num = 1
    sample_num = 10
    for data in ds_test.create_tuple_iterator(output_numpy=True):
        i += 1
        images = data[0].astype(np.float32)
        target_features = net.get_layer_output(Tensor(images), 'conv11')[:sample_num]
        if i >= batch_num:
            break

    # run attacking
    model_inversion_attack = ModelInversionAttack(net, inv_net, input_shape=(3, 32, 32), split_layer='conv11')
    model_inversion_attack.train_inversion_model(ds, epochs=100)

    inversion_images = model_inversion_attack.inverse(target_features).asnumpy()
    inversion_images = inversion_images.clip(0, 1)
    # evaluate the quality of inversion images
    avg_ssim, avg_psnr = model_inversion_attack.evaluate(ds_test)
    LOGGER.info(TAG, 'The average ssim value between original images and inverted images is: {}'.format(avg_ssim))
    LOGGER.info(TAG, 'The average psnr value between original images and inverted images is: {}'.format(avg_psnr))

    # plot 10 images
    plot_num = min(sample_num, 10)
    for n in range(1, plot_num+1):
        plt.subplot(2, plot_num, n)
        if n == 1:
            plt.title('Original images', fontsize=16, loc='left')
        plt.imshow(images[n - 1].transpose(1, 2, 0))
        plt.subplot(2, plot_num, n + plot_num)
        if n == 1:
            plt.title('Inverted images', fontsize=16, loc='left')
        plt.gray()
        plt.imshow(inversion_images[n - 1].transpose(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    # device_target can be "CPU", "GPU" or "Ascend"
    context.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    ckpt_path = '../../common/networks/cifar10cnn/trained_ckpt_file/checkpoint_cifar-10_1562.ckpt'
    cifar_inversion_attack(CIFAR10CNN(), CIFAR10CNNDecoderConv11(), ckpt_path)
