# Copyright 2022 Huawei Technologies Co., Ltd
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

"""Pass in the arguments and execute the corresponding attack algorithms."""

import copy
import argparse
import os
from launch_attack import run_attack


def arg_parse():
    """parse the arguments."""
    parse = argparse.ArgumentParser()
    parse.add_argument('--out_put', type=str, default='output/', help='directory for save outputs')
    parse.add_argument('--data_path', type=str, default='./data', help='path of dataset')
    parse.add_argument('--dataset', type=str, default="WebImage", choices=['TinyImageNet', 'CIFAR100', 'WebImage'])
    parse.add_argument('--model', type=str, default="resnet34", choices=['resnet18', 'resnet34'])
    parse.add_argument('--alg_name', type=str, default="StepWise", choices=['InvGrad', 'SeeThrough', 'StepWise'])
    parse.add_argument('--defense', type=str, default="None", choices=['None', 'Vicinal Augment',
                                                                       'Differential Privacy', 'Gradient Prune'])
    parse.add_argument('--num_data_points', type=int, default=4)
    parse.add_argument('--max_iterations', type=int, default=7000)
    parse.add_argument('--step_size', type=float, default=0.1)
    parse.add_argument('--TV_scale', type=float, default=0.002)
    parse.add_argument('--TV_start', type=int, default=2000)
    parse.add_argument('--BN_scale', type=float, default=0.0001)
    parse.add_argument('--BN_start', type=int, default=3000)
    parse.add_argument('--callback', type=int, default=100)

    args_ = parse.parse_args()
    args_.eva_txt_file = 'resultTXT.txt'

    return args_


def run_main(args_):
    arg_dic = copy.deepcopy(vars(args_))
    arg_dic['custom_parameter'] = {}
    for key in vars(args_):
        arg_dic['custom_parameter'][key] = getattr(args_, key)
    if not os.path.exists(arg_dic['out_put']):
        os.mkdir(arg_dic['out_put'])
    run_attack(arg_dic)
    print('Running over! Check the output in directory: ' + arg_dic['out_put'])


if __name__ == '__main__':
    args = arg_parse()
    run_main(args)
