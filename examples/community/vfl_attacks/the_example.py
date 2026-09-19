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
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/utils/')
import mindspore
from utils.vfl.init_vfl import Init_Vfl
from utils.vfl.vfl import VFL
from utils.methods.direct_attack.direct_attack_vfl import DirectVFL
from utils.config.args_process import argsments_function
from examples.model.init_active_model import init_top_model
from examples.model.init_passive_model import init_bottom_model
from examples.datasets.cifar_dataset import get_cifar_dataloader
mindspore.set_context(mode=mindspore.GRAPH_MODE)

if __name__ == '__main__':
    gpu = 0
    if gpu:
        mindspore.set_context(device_target="GPU")
    epochs = 100
    batch_size = 32
    lr = 0.005
    top_model_trainable = 0
    n_party = 2
    adversary = 1
    num_classes = 10
    topk = 1
    target_train_size = -1
    target_test_size = -1
    backdoor_test_size = 2000
    # Support g_r and direct_attack.
    alg = 'direct_attack'
    backdoor_label = 6
    poison_num = 500
    # Process the arguments.
    args = argsments_function(epochs, batch_size, lr, top_model_trainable, n_party, adversary, gpu,
                               target_train_size, target_test_size, backdoor_test_size, backdoor_label,
                               alg, num_classes, poison_num, topk)
    local_args = args.copy()
    local_args['dataset'] = 'cifar10'
    local_args['half'] = 16
    local_args['model_type'] = 'Resnet'
    # Define dataset loader.
    train_dl, test_dl, backdoor_test_dl, backdoor_indices, backdoor_target_indices = get_cifar_dataloader(local_args)
    # Define models.
    bottoms = [init_bottom_model('active', local_args)]
    for i in range(0, args['n_passive_party']):
        passive_party_model = init_bottom_model('passive', local_args)
        bottoms.append(passive_party_model)
    top = init_top_model(local_args)
    # Initialize vfl framework.
    if alg == 'g_r':
        init_vfl = Init_Vfl(args)
        init_vfl.get_vfl(bottoms, top, train_dl, backdoor_target_indices, backdoor_indices)
        VFL_framework = VFL(train_dl, test_dl, init_vfl, backdoor_test_dl)
        VFL_framework.train()
    elif alg == 'direct_attack':
        init_vfl = Init_Vfl(args)
        init_vfl.get_vfl(bottoms, top, train_dl)
        VFL_framework = DirectVFL(train_dl, test_dl, init_vfl)
        VFL_framework.train()