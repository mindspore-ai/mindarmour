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


"""
Fault injection example.
Download checkpoint from: https://www.mindspore.cn/resources/hub or just trained your own checkpoint.
Download dataset from: http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz.
File structure:
    --cifar10-batches-bin
        --train
            --data_batch_1.bin
            --data_batch_2.bin
            --data_batch_3.bin
            --data_batch_4.bin
            --data_batch_5.bin
        --test
            --test_batch.bin

Please extract and restructure the file as shown above.
"""
import argparse

from mindspore import Model, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindarmour.reliability.model_fault_injection.fault_injection import FaultInjector
from examples.common.networks.lenet5.lenet5_net import LeNet5
from examples.common.networks.vgg.vgg import vgg16
from examples.common.networks.resnet.resnet import resnet50
from examples.common.dataset.data_processing import create_dataset_cifar, generate_mnist_dataset


parser = argparse.ArgumentParser(description='layer_states')
parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'])
parser.add_argument('--model', type=str, default='lenet', choices=['lenet', 'resnet50', 'vgg16'])
parser.add_argument('--device_id', type=int, default=0)
args = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)


test_flag = args.model
if test_flag == 'lenet':
    # load data
    DATA_FILE = '../common/dataset/MNIST_Data/test'
    ckpt_path = '../common/networks/lenet5/trained_ckpt_file/checkpoint_lenet-10_1875.ckpt'
    ds_eval = generate_mnist_dataset(DATA_FILE, batch_size=64)
    net = LeNet5()
elif test_flag == 'vgg16':
    from examples.common.networks.vgg.config import cifar_cfg as cfg
    DATA_FILE = '../common/dataset/cifar10-batches-bin'
    ckpt_path = '../common/networks/vgg16_ascend_v111_cifar10_offical_cv_bs64_acc93.ckpt'
    ds_eval = create_dataset_cifar(DATA_FILE, 224, 224, training=False)
    net = vgg16(10, cfg, 'test')
elif test_flag == 'resnet50':
    DATA_FILE = '../common/dataset/cifar10-batches-bin'
    ckpt_path = '../common/networks/resnet50_ascend_v111_cifar10_offical_cv_bs32_acc92.ckpt'
    ds_eval = create_dataset_cifar(DATA_FILE, 224, 224, training=False)
    net = resnet50(10)
else:
    exit()
param_dict = load_checkpoint(ckpt_path)
load_param_into_net(net, param_dict)
model = Model(net)

# Initialization
fi_type = ['bitflips_random', 'bitflips_designated', 'random', 'zeros',
           'nan', 'inf', 'anti_activation', 'precision_loss']
fi_mode = ['single_layer', 'all_layer']
fi_size = [1, 2, 3]

# Fault injection
fi = FaultInjector(model, ds_eval, fi_type, fi_mode, fi_size)
results = fi.kick_off()
result_summary = fi.metrics()

# print result
for result in results:
    print(result)
for result in result_summary:
    print(result)
