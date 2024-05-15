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
Examples of shadow model attack
"""
import numpy as np
import matplotlib.pyplot as plt

import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Tensor, context
from mindarmour.privacy.evaluation.inversion_attack import ImageInversionAttack
from mindarmour.privacy.evaluation.shadow_model_attack import ShadowModelAttack
from mindarmour.utils.logger import LogUtil

from examples.common.networks.cifar10cnn.cifar10cnn_net import CIFAR10CNN
from examples.privacy.inversion_attack.shadow_net import CIFAR10CNNAlternativeConv11Arch2
from examples.common.dataset.data_processing import generate_dataset_cifar
LOGGER = LogUtil.get_instance()
LOGGER.set_level('INFO')
TAG = 'ShadowModelAttack'


# pylint: disable=invalid-name
def cifar_inversion_attack(net, shadow_net, ckptpath):
    """
    Image inversion attack based on CNN and CAFIR10 dataset.
    """
    # upload trained network
    load_dict = load_checkpoint(ckptpath)
    load_param_into_net(net, load_dict)

    # get original data and their inferred fearures
    data_list = "../../common/dataset/CIFAR10/train"
    ds = generate_dataset_cifar(data_list, 32, repeat_num=1)
    data_list = "../../common/dataset/CIFAR10/test"
    ds_test = generate_dataset_cifar(data_list, 32, repeat_num=1)
    i = 0
    batch_num = 1
    sample_num = 10

    # run attacking
    shadow_model_attack = ShadowModelAttack(net, shadow_net, split_layer='conv11')
    shadow_model_attack.train_shadow_model(ds, attack_config={'epochs': 100})
    shadow_model_attack.evaluate(ds_test, 10)
    for data in ds_test.create_tuple_iterator(output_numpy=True):
        i += 1
        images = data[0].astype(np.float32)
        target_features = shadow_net.getlayeroutput(Tensor(images), 'conv11').asnumpy()[:sample_num]
        original_images = images[: sample_num]
        if i >= batch_num:
            break

    inversion_attack = ImageInversionAttack(shadow_net, (3, 32, 32), input_bound=(0, 1), loss_weights=(1, 0.1, 5))
    inversion_images = inversion_attack.generate(target_features, iters=150)
    # evaluate the quality of inversion images
    avg_l2_dis, avg_ssim, _ = inversion_attack.evaluate(original_images, inversion_images)
    LOGGER.info(TAG, 'The average L2 distance between original images and inverted images is: {}'.format(avg_l2_dis))
    LOGGER.info(TAG, 'The average ssim value between original images and inverted images is: {}'.format(avg_ssim))

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
        plt.imshow(inversion_images[n - 1].transpose(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    # device_target can be "CPU", "GPU" or "Ascend"
    context.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")
    ckpt_path = 'examples/common/networks/cifar10cnn/trained_ckpt_file/checkpoint_cifar-10_1562.ckpt'
    cifar_inversion_attack(CIFAR10CNN(), CIFAR10CNNAlternativeConv11Arch2(), ckpt_path)
