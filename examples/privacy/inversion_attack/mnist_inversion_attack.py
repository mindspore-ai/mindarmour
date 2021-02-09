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
"""
Examples of image inversion attack
"""
import numpy as np
import matplotlib.pyplot as plt

from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Tensor, context
from mindspore import nn
from mindarmour.privacy.evaluation.inversion_attack import ImageInversionAttack
from mindarmour.utils.logger import LogUtil

from examples.common.networks.lenet5.lenet5_net import LeNet5, conv, fc_with_initialize
from examples.common.dataset.data_processing import generate_mnist_dataset

LOGGER = LogUtil.get_instance()
LOGGER.set_level('INFO')
TAG = 'InversionAttack'


# pylint: disable=invalid-name
class LeNet5_part(nn.Cell):
    """
    Part of LeNet5 network.
    """
    def __init__(self):
        super(LeNet5_part, self).__init__()
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16*5*5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        return x


def mnist_inversion_attack(net):
    """
    Image inversion attack based on LeNet5 and MNIST dataset.
    """
    # upload trained network
    ckpt_path = '../../common/networks/lenet5/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
    load_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, load_dict)

    # get original data and their inferred fearures
    data_list = "../../common/dataset/MNIST/train"
    batch_size = 32
    ds = generate_mnist_dataset(data_list, batch_size)
    i = 0
    batch_num = 1
    sample_num = 30
    for data in ds.create_tuple_iterator(output_numpy=True):
        i += 1
        images = data[0].astype(np.float32)
        true_labels = data[1][: sample_num]
        target_features = net(Tensor(images)).asnumpy()[:sample_num]
        original_images = images[: sample_num]
        if i >= batch_num:
            break

    # run attacking
    inversion_attack = ImageInversionAttack(net, input_shape=(1, 32, 32), input_bound=(0, 1), loss_weights=[1, 0.1, 5])
    inversion_images = inversion_attack.generate(target_features, iters=100)

    # get the predict results of inversion images on a new trained model
    net2 = LeNet5()
    new_ckpt_path = '../../common/networks/lenet5/new_trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
    new_load_dict = load_checkpoint(new_ckpt_path)
    load_param_into_net(net2, new_load_dict)
    pred_labels = np.argmax(net2(Tensor(inversion_images).astype(np.float32)).asnumpy(), axis=1)

    # evaluate the quality of inversion images
    avg_l2_dis, avg_ssim, avg_confi = inversion_attack.evaluate(original_images, inversion_images, true_labels, net2)
    LOGGER.info(TAG, 'The average L2 distance between original images and inverted images is: {}'.format(avg_l2_dis))
    LOGGER.info(TAG, 'The average ssim value between original images and inverted images is: {}'.format(avg_ssim))
    LOGGER.info(TAG, 'The average prediction confidence on true labels of inverted images is: {}'.format(avg_confi))
    LOGGER.info(TAG, 'True labels of original images are:      %s' % true_labels)
    LOGGER.info(TAG, 'Predicted labels of inverted images are: %s' % pred_labels)

    # plot 10 images
    plot_num = min(sample_num, 10)
    for n in range(1, plot_num+1):
        plt.subplot(2, plot_num, n)
        if n == 1:
            plt.title('Original images', fontsize=16, loc='left')
        plt.gray()
        plt.imshow(images[n - 1].reshape(32, 32))
        plt.subplot(2, plot_num, n + plot_num)
        if n == 1:
            plt.title('Inverted images', fontsize=16, loc='left')
        plt.gray()
        plt.imshow(inversion_images[n - 1].reshape(32, 32))
    plt.show()


if __name__ == '__main__':
    # device_target can be "CPU", "GPU" or "Ascend"
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    # attack based on complete LeNet5
    mnist_inversion_attack(LeNet5())
    # attack based on part of LeNet5. The network is more shallower and can lead to a better attack result
    mnist_inversion_attack(LeNet5_part())
