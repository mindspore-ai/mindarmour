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
An example of fuzz testing and then enhance non-robustness model.
"""
import random
import numpy as np

import mindspore
from mindspore import Model
from mindspore import context
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim.momentum import Momentum

from mindarmour.adv_robustness.defenses import AdversarialDefense
from mindarmour.fuzz_testing import Fuzzer
from mindarmour.fuzz_testing import KMultisectionNeuronCoverage
from mindarmour.utils.logger import LogUtil

from examples.common.dataset.data_processing import generate_mnist_dataset
from examples.common.networks.lenet5.lenet5_net_for_fuzzing import LeNet5

LOGGER = LogUtil.get_instance()
TAG = 'Fuzz_testing and enhance model'
LOGGER.set_level('INFO')


def split_dataset(image, label, proportion):
    """
    Split the generated fuzz data into train and test set.
    """
    indices = np.arange(len(image))
    random.shuffle(indices)
    train_length = int(len(image) * proportion)
    train_image = [image[i] for i in indices[:train_length]]
    train_label = [label[i] for i in indices[:train_length]]
    test_image = [image[i] for i in indices[:train_length]]
    test_label = [label[i] for i in indices[:train_length]]
    return train_image, train_label, test_image, test_label


def example_lenet_mnist_fuzzing():
    """
    An example of fuzz testing and then enhance the non-robustness model.
    """
    # upload trained network
    ckpt_path = '../common/networks/lenet5/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
    net = LeNet5()
    load_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, load_dict)
    model = Model(net)
    mutate_config = [
        {'method': 'GaussianBlur',
         'params': {'ksize': [1, 2, 3, 5], 'auto_param': [True, False]}},
        {'method': 'MotionBlur',
         'params': {'degree': [1, 2, 5], 'angle': [45, 10, 100, 140, 210, 270, 300], 'auto_param': [True]}},
        {'method': 'GradientBlur',
         'params': {'point': [[10, 10]], 'auto_param': [True]}},
        {'method': 'UniformNoise',
         'params': {'factor': [0.1, 0.2, 0.3], 'auto_param': [False, True]}},
        {'method': 'GaussianNoise',
         'params': {'factor': [0.1, 0.2, 0.3], 'auto_param': [False, True]}},
        {'method': 'SaltAndPepperNoise',
         'params': {'factor': [0.1, 0.2, 0.3], 'auto_param': [False, True]}},
        {'method': 'NaturalNoise',
         'params': {'ratio': [0.1], 'k_x_range': [(1, 3), (1, 5)], 'k_y_range': [(1, 5)], 'auto_param': [False, True]}},
        {'method': 'Contrast',
         'params': {'alpha': [0.5, 1, 1.5], 'beta': [-10, 0, 10], 'auto_param': [False, True]}},
        {'method': 'GradientLuminance',
         'params': {'color_start': [(0, 0, 0)], 'color_end': [(255, 255, 255)], 'start_point': [(10, 10)],
                    'scope': [0.5], 'pattern': ['light'], 'bright_rate': [0.3], 'mode': ['circle'],
                    'auto_param': [False, True]}},
        {'method': 'Translate',
         'params': {'x_bias': [0, 0.05, -0.05], 'y_bias': [0, -0.05, 0.05], 'auto_param': [False, True]}},
        {'method': 'Scale',
         'params': {'factor_x': [1, 0.9], 'factor_y': [1, 0.9], 'auto_param': [False, True]}},
        {'method': 'Shear',
         'params': {'factor': [0.2, 0.1], 'direction': ['horizontal', 'vertical'], 'auto_param': [False, True]}},
        {'method': 'Rotate',
         'params': {'angle': [20, 90], 'auto_param': [False, True]}},
        {'method': 'Perspective',
         'params': {'ori_pos': [[[0, 0], [0, 800], [800, 0], [800, 800]]],
                    'dst_pos': [[[50, 0], [0, 800], [780, 0], [800, 800]]], 'auto_param': [False, True]}},
        {'method': 'Curve',
         'params': {'curves': [5], 'depth': [2], 'mode': ['vertical'], 'auto_param': [False, True]}},
        {'method': 'FGSM',
         'params': {'eps': [0.3, 0.2, 0.4], 'alpha': [0.1], 'bounds': [(0, 1)]}}]

    # get training data
    data_list = "../common/dataset/MNIST/train"
    batch_size = 32
    ds = generate_mnist_dataset(data_list, batch_size, sparse=False)
    train_images = []
    for data in ds.create_tuple_iterator(output_numpy=True):
        images = data[0].astype(np.float32)
        train_images.append(images)
    train_images = np.concatenate(train_images, axis=0)
    segmented_num = 100

    # fuzz test with original test data
    data_list = "../common/dataset/MNIST/test"
    batch_size = batch_size
    init_samples = 50
    max_iters = 500
    mutate_num_per_seed = 10
    ds = generate_mnist_dataset(data_list, batch_size=batch_size, num_samples=init_samples, sparse=False)
    test_images = []
    test_labels = []
    for data in ds.create_tuple_iterator(output_numpy=True):
        test_images.append(data[0].astype(np.float32))
        test_labels.append(data[1])
    test_images = np.concatenate(test_images, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    coverage = KMultisectionNeuronCoverage(model, train_images, segmented_num=segmented_num, incremental=True)
    kmnc = coverage.get_metrics(test_images[:100])
    print('kmnc: ', kmnc)

    # make initial seeds
    initial_seeds = []
    for img, label in zip(test_images, test_labels):
        initial_seeds.append([img, label])

    model_fuzz_test = Fuzzer(model)
    gen_samples, gt, _, _, metrics = model_fuzz_test.fuzzing(mutate_config,
                                                             initial_seeds, coverage,
                                                             evaluate=True,
                                                             max_iters=max_iters,
                                                             mutate_num_per_seed=mutate_num_per_seed)

    if metrics:
        for key in metrics:
            LOGGER.info(TAG, key + ': %s', metrics[key])

    train_image, train_label, test_image, test_label = split_dataset(gen_samples, gt, 0.7)

    # load model B and test it on the test set
    ckpt_path = '../common/networks/lenet5/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
    net = LeNet5()
    load_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, load_dict)
    model_b = Model(net)
    pred_b = model_b.predict(Tensor(test_image, dtype=mindspore.float32)).asnumpy()
    acc_b = np.sum(np.argmax(pred_b, axis=1) == np.argmax(test_label, axis=1)) / len(test_label)
    print('Accuracy of model B on test set is ', acc_b)

    # enhense model robustness
    lr = 0.001
    momentum = 0.9
    loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True)
    optimizer = Momentum(net.trainable_params(), lr, momentum)

    adv_defense = AdversarialDefense(net, loss_fn, optimizer)
    adv_defense.batch_defense(np.array(train_image).astype(np.float32), np.argmax(train_label, axis=1).astype(np.int32))
    preds_en = net(Tensor(test_image, dtype=mindspore.float32)).asnumpy()
    acc_en = np.sum(np.argmax(preds_en, axis=1) == np.argmax(test_label, axis=1)) / len(test_label)
    print('Accuracy of enhensed model on test set is ', acc_en)


if __name__ == '__main__':
    # device_target can be "CPU", "GPU" or "Ascend"
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    example_lenet_mnist_fuzzing()
