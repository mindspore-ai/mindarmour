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
import numpy as np
from mindspore import Model
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindarmour.fuzz_testing.model_coverage_metrics import NeuronCoverage, TopKNeuronCoverage, NeuronBoundsCoverage,\
    SuperNeuronActivateCoverage, KMultisectionNeuronCoverage
from mindarmour.utils.logger import LogUtil

from examples.common.dataset.data_processing import generate_mnist_dataset
from examples.common.networks.lenet5.lenet5_net_for_fuzzing import LeNet5

LOGGER = LogUtil.get_instance()
TAG = 'Neuron coverage test'
LOGGER.set_level('INFO')


def test_lenet_mnist_coverage():
    # upload trained network
    ckpt_path = '../common/networks/lenet5/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
    net = LeNet5()
    load_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, load_dict)
    model = Model(net)

    # get training data
    data_list = "../common/dataset/MNIST/train"
    batch_size = 32
    ds = generate_mnist_dataset(data_list, batch_size, sparse=True)
    train_images = []
    for data in ds.create_tuple_iterator(output_numpy=True):
        images = data[0].astype(np.float32)
        train_images.append(images)
    train_images = np.concatenate(train_images, axis=0)

    # fuzz test with original test data
    # get test data
    data_list = "../common/dataset/MNIST/test"
    batch_size = 32
    ds = generate_mnist_dataset(data_list, batch_size, sparse=True)
    test_images = []
    test_labels = []
    for data in ds.create_tuple_iterator(output_numpy=True):
        images = data[0].astype(np.float32)
        labels = data[1]
        test_images.append(images)
        test_labels.append(labels)
    test_images = np.concatenate(test_images, axis=0)

    # initialize fuzz test with training dataset
    nc = NeuronCoverage(model, threshold=0.1)
    nc_metric = nc.get_metrics(test_images)

    tknc = TopKNeuronCoverage(model, top_k=3)
    tknc_metrics = tknc.get_metrics(test_images)

    snac = SuperNeuronActivateCoverage(model, train_images)
    snac_metrics = snac.get_metrics(test_images)

    nbc = NeuronBoundsCoverage(model, train_images)
    nbc_metrics = nbc.get_metrics(test_images)

    kmnc = KMultisectionNeuronCoverage(model, train_images, segmented_num=100)
    kmnc_metrics = kmnc.get_metrics(test_images)

    print('KMNC of this test is: ', kmnc_metrics)
    print('NBC of this test is: ', nbc_metrics)
    print('SNAC of this test is: ', snac_metrics)
    print('NC of this test is: ', nc_metric)
    print('TKNC of this test is: ', tknc_metrics)


if __name__ == '__main__':
    # device_target can be "CPU", "GPU" or "Ascend"
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_lenet_mnist_coverage()
