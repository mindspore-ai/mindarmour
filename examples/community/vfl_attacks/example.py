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
import argparse
import ast
import constant
import mindspore as ms
constant.init()

def user_initial_selection():
    args = {}
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', type=str, default='pytorch')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--step-gamma', type=float, default=0.1)
    parser.add_argument('--attack', type=ast.literal_eval, default=False)
    parser.add_argument('--backdoor', type=str, default='no', choices=['no', 'lr_ba', 'g_r', 'villain'])
    parser.add_argument('--label-inference-attack', type=str, default='no',
                        choices=['no', 'passive_model_completion', 'active_model_completion', 'direct_attack'])

    temp = parser.parse_args()
    args['config'] = 'evaluate/config/' + temp.dataset + '.yaml'
    args['framework'] = temp.framework
    args['cuda'] = temp.cuda
    args['batch_size'] = temp.batch_size
    args['epochs'] = temp.epochs
    args['lr'] = temp.lr
    args['step_gamma'] = temp.step_gamma
    args['attack'] = temp.attack
    args['backdoor'] = temp.backdoor
    args['label_inference_attack'] = temp.label_inference_attack
    args['n_party'] = 2
    args['top_model'] = True
    args['aggregate'] = 'Concate'
    args['poison_rate'] = 0.01
    if temp.dataset == 'cifar100':
        args['poison_rate'] = 0.001
    return args

def get_cfg(input_args):
        ms.set_context(env_config_path='configure.json')
        from evaluate.args_line import run_mindspore
        run_mindspore(input_args)

if __name__ == '__main__':
    input_args = user_initial_selection()
    get_cfg(input_args)
