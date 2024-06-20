# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""an example of CryptoSFL using MLP as the model"""
import copy

import numpy as np
import mindspore
import mindspore.dataset as ds
import mindspore.nn as mspnn
import mindspore.dtype as mstype
from mindspore import ParameterTuple, ops
from mindspore.dataset import vision, transforms

from crypto.sife_dynamic import SIFEDynamicClient
from crypto.sife_dynamic import SIFEDynamicTPA
from crypto.utils import load_dlog_table_config
from nn.smc import Secure2PCClient
from nn.smc import Secure2PCServer
from nn.utils import initiate_server_model
from nn.worker import CryptoClient
from nn.worker import CryptoServer


def dataset_distribute(num_users):
    """Distribute dataset among users."""
    num_items = int(60000 / num_users)
    dict_users, all_indices = {}, [i for i in range(60000)]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_indices, num_items, replace=False))
        all_indices = list(set(all_indices) - dict_users[i])
    return dict_users


def datapipe(path, batch_size, sampler):
    """Prepare data pipeline for training."""
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(mindspore.int32)

    dataset = ds.MnistDataset(path, sampler=sampler)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset


def get_dataloaders(num_users, batch_size):
    """Get dataloaders for training and testing."""
    dict_users = dataset_distribute(num_users)
    train_loader_dict = {}
    for i in range(num_users):
        train_loader_dict[i] = datapipe(path='./datasets/MNIST_Data/train', batch_size=batch_size,
                                        sampler=list(dict_users[i]))
    test_loader = datapipe(path='./datasets/MNIST_Data/test', batch_size=batch_size, sampler=list(range(10000)))
    return train_loader_dict, test_loader


def aggregate_weights(w_locals_client, w_locals_server):
    """Aggregate weights from multiple clients."""
    num_users = len(w_locals_client)
    # server weights aggregation
    w_avg_server = copy.deepcopy(w_locals_server[0])
    for k in range(len(w_avg_server)):
        for i in range(1, len(w_locals_server)):
            w_avg_server[k] += w_locals_server[i][k]
        w_avg_server[k] = np.divide(w_avg_server[k], len(w_locals_server))

    # client weights aggregation
    w_avg_client = copy.deepcopy(w_locals_client[0])
    for k in w_avg_client.keys():
        for i in range(1, len(w_locals_client)):
            w_avg_client[k] += w_locals_client[i][k]
        w_avg_client[k] = ops.div(w_avg_client[k], len(w_locals_client))

    for key, value in w_avg_client.items():
        w_avg_client[key] = mindspore.Parameter(value.asnumpy())

    return [copy.deepcopy(w_avg_client) for _ in range(num_users)], \
           [copy.deepcopy(w_avg_server) for _ in range(num_users)]


def initiate_model_list(lr, model, precision_data=3, precision_weight=3):
    """Initialize model list for clients and server."""
    # initiate the crypto system
    sec_param_config_file = './config/sec_param.json'
    dlog_table_config_file = './config/dlog_b8.json'
    eta = 1250
    sec_param = 256
    dlog = load_dlog_table_config(dlog_table_config_file)
    sife_tpa = SIFEDynamicTPA(eta, sec_param=sec_param, sec_param_config=sec_param_config_file)
    sife_tpa.setup()
    sife_enc_client = SIFEDynamicClient(role='enc')
    sife_dec_client = SIFEDynamicClient(role='dec', dlog=dlog)

    secure2pc_client = Secure2PCClient(sife=(sife_tpa, sife_enc_client), precision=precision_data)
    secure2pc_server = Secure2PCServer(sife=(sife_tpa, sife_dec_client), precision=(precision_data, precision_weight))

    n_features, hidden_layers = initiate_server_model(model)

    client = CryptoClient(smc=secure2pc_client, model=model)
    server = CryptoServer(n_features=n_features, hidden_layers=hidden_layers, eta=lr, smc=secure2pc_server)
    return client, server


class GradNet(mspnn.Cell):
    """Gradient network for training."""

    def __init__(self, net, grad_wrt_output: np.ndarray):
        super(GradNet, self).__init__()
        self.net = net
        self.params = ParameterTuple(net.trainable_params())
        self.grad_op = ops.GradOperation(get_by_list=True, sens_param=True)

        # grads of the smashed data
        self.grad_wrt_output = mindspore.Tensor(grad_wrt_output, dtype=mstype.float32)

    def construct(self, x):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(x, self.grad_wrt_output)


def train_and_test(epochs, num_users, batch_size, model, lr):
    """Train and test the model."""
    # set client & server model
    client, server = initiate_model_list(lr, model)

    w_locals_client = [copy.deepcopy(client.parameters_dict()) for _ in range(num_users)]
    w_locals_server = [copy.deepcopy(server.w) for _ in range(num_users)]

    # load data
    train_loader_dict, test_loader = get_dataloaders(num_users, batch_size)

    for epoch in range(epochs):
        # training
        for idx in range(num_users):
            if epoch != 0:
                _, _ = mindspore.load_param_into_net(client, w_locals_client[idx])

            server.w = copy.deepcopy(w_locals_server[idx])
            optimizer = mspnn.SGD(client.trainable_params(), learning_rate=lr)
            for _, (inputs, targets) in enumerate(train_loader_dict[idx].create_tuple_iterator()):
                # client forward
                plain_intermediate = client(inputs)
                ct_feedforward, ct_backpropagation, y_onehot = client.encrypt(
                    intermediate=plain_intermediate.asnumpy(), y=targets.asnumpy())
                # server forward
                z, a = server.feedforward_secure(ct_batch=ct_feedforward, w=copy.deepcopy(server.w))
                # server backward
                grad, d_intermediate = server.get_gradient_secure(ct_batch=ct_backpropagation, y_encode=y_onehot,
                                                                  a=copy.deepcopy(a), z=copy.deepcopy(z),
                                                                  w=copy.deepcopy(server.w))
                delta_w = [server.eta * grad[i] for i in range(len(server.w))]
                for i in range(len(server.w)):
                    server.w[i] -= delta_w[i]
                # client backward
                client_grads = GradNet(net=client, grad_wrt_output=d_intermediate)(inputs)
                optimizer(client_grads)

            w_locals_client[idx] = copy.deepcopy(client.parameters_dict())
            w_locals_server[idx] = copy.deepcopy(server.w)

        # aggregate weights after each global epoch
        w_locals_client, w_locals_server = aggregate_weights(w_locals_client, w_locals_server)

        # testing
        _, _ = mindspore.load_param_into_net(client, w_locals_client[0])
        server.w = copy.deepcopy(w_locals_server[0])
        correct, total = 0, 0

        for inputs, targets in test_loader.create_tuple_iterator():
            plain_intermediate = client(inputs)
            pred = server.predict(plain_intermediate)
            correct += np.sum(targets.asnumpy() == pred, axis=0)
            total += len(targets)
        test_accuracy = correct / total
        print('epoch {}, test accuracy = {:.2f}%'.format(epoch + 1, 100 * test_accuracy))


if __name__ == "__main__":
    rounds = 20
    user_counts = 5
    bs = 50
    client_net = 'mlp64'
    learning_rate = 1e-3
    train_and_test(rounds, user_counts, bs, client_net, learning_rate)
