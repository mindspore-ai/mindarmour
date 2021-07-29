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
from mindarmour.fuzz_testing import ModelCoverageMetrics
from mindarmour.utils.logger import LogUtil

from examples.common.dataset.data_processing import generate_mnist_dataset
from examples.common.networks.lenet5.lenet5_net_for_fuzzing import LeNet5

LOGGER = LogUtil.get_instance()
TAG = 'Fuzz_testing and enhance model'
LOGGER.set_level('INFO')


def example_lenet_mnist_fuzzing():
    """
    An example of fuzz testing and then enhance the non-robustness model.
    """
    # upload trained network
    ckpt_path = '../common/networks/lenet5/trained_ckpt_file/lenet_m1-10_1250.ckpt'
    net = LeNet5()
    load_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, load_dict)
    model = Model(net)
    mutate_config = [{'method': 'Blur',
                      'params': {'auto_param': [True]}},
                     {'method': 'Contrast',
                      'params': {'auto_param': [True]}},
                     {'method': 'Translate',
                      'params': {'auto_param': [True]}},
                     {'method': 'Brightness',
                      'params': {'auto_param': [True]}},
                     {'method': 'Noise',
                      'params': {'auto_param': [True]}},
                     {'method': 'Scale',
                      'params': {'auto_param': [True]}},
                     {'method': 'Shear',
                      'params': {'auto_param': [True]}},
                     {'method': 'FGSM',
                      'params': {'eps': [0.3, 0.2, 0.4], 'alpha': [0.1]}}
                     ]

    # get training data
    data_list = "../common/dataset/MNIST/train"
    batch_size = 32
    ds = generate_mnist_dataset(data_list, batch_size, sparse=False)
    train_images = []
    for data in ds.create_tuple_iterator(output_numpy=True):
        images = data[0].astype(np.float32)
        train_images.append(images)
    train_images = np.concatenate(train_images, axis=0)
    neuron_num = 10
    segmented_num = 1000

    # initialize fuzz test with training dataset
    model_coverage_test = ModelCoverageMetrics(model, neuron_num, segmented_num, train_images)

    # fuzz test with original test data
    # get test data
    data_list = "../common/dataset/MNIST/test"
    batch_size = 32
    init_samples = 5000
    max_iters = 50000
    mutate_num_per_seed = 10
    ds = generate_mnist_dataset(data_list, batch_size, num_samples=init_samples,
                                sparse=False)
    test_images = []
    test_labels = []
    for data in ds.create_tuple_iterator(output_numpy=True):
        images = data[0].astype(np.float32)
        labels = data[1]
        test_images.append(images)
        test_labels.append(labels)
    test_images = np.concatenate(test_images, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    initial_seeds = []

    # make initial seeds
    for img, label in zip(test_images, test_labels):
        initial_seeds.append([img, label])

    model_coverage_test.calculate_coverage(
        np.array(test_images[:100]).astype(np.float32))
    LOGGER.info(TAG, 'KMNC of test dataset before fuzzing is : %s',
                model_coverage_test.get_kmnc())
    LOGGER.info(TAG, 'NBC of test dataset before fuzzing is : %s',
                model_coverage_test.get_nbc())
    LOGGER.info(TAG, 'SNAC of test dataset before fuzzing is : %s',
                model_coverage_test.get_snac())

    model_fuzz_test = Fuzzer(model, train_images, 10, 1000)
    gen_samples, gt, _, _, metrics = model_fuzz_test.fuzzing(mutate_config,
                                                             initial_seeds,
                                                             eval_metrics='auto',
                                                             max_iters=max_iters,
                                                             mutate_num_per_seed=mutate_num_per_seed)

    if metrics:
        for key in metrics:
            LOGGER.info(TAG, key + ': %s', metrics[key])

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

    train_image, train_label, test_image, test_label = split_dataset(
        gen_samples, gt, 0.7)

    # load model B and test it on the test set
    ckpt_path = '../common/networks/lenet5/trained_ckpt_file/lenet_m2-10_1250.ckpt'
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
    loss_fn = SoftmaxCrossEntropyWithLogits(Sparse=True)
    optimizer = Momentum(net.trainable_params(), lr, momentum)

    adv_defense = AdversarialDefense(net, loss_fn, optimizer)
    adv_defense.batch_defense(np.array(train_image).astype(np.float32),
                              np.argmax(train_label, axis=1).astype(np.int32))
    preds_en = net(Tensor(test_image, dtype=mindspore.float32)).asnumpy()
    acc_en = np.sum(np.argmax(preds_en, axis=1) == np.argmax(test_label, axis=1)) / len(test_label)
    print('Accuracy of enhensed model on test set is ', acc_en)


if __name__ == '__main__':
    # device_target can be "CPU", "GPU" or "Ascend"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    example_lenet_mnist_fuzzing()
