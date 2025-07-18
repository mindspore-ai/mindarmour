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
Parse configuration file
"""

import logging
import yaml
from MindsporeCode.datasets.base_dataset import get_num_classes

def get_args(file, file_time):
    """
    parse configuration yaml file

    :return: configuration
    """
    yaml.warnings({'YAMLLoadWarning': False})
    f = open(file, 'r', encoding='utf-8')
    cfg = f.read()
    args = yaml.load(cfg, Loader=yaml.SafeLoader)
    f.close()
    args['num_classes'] = get_num_classes(args['dataset'])

    if 'train_label_non_iid' not in args.keys():
        args['train_label_non_iid'] = None
    if 'train_label_fix_backdoor' not in args.keys():
        args['train_label_fix_backdoor'] = -1

    # the configuration whether to print the execution time of federated training and LR-BA
    args['time'] = False

    time = file_time
    set_logging(args['log'], time)
    return args


def set_logging(log_file, time):
    """
    configure logging INFO messaged located in tests/result

    :param str log_file: path of log file
    """
    # file_name = '../temp_output/{}-{}.txt'.format(log_file, time)
    # constant.set_value('file_name', file_name)
    logging.basicConfig(
        level=logging.INFO,
        filename='../temp_output/{}-{}.txt'.format(log_file, time),
        filemode='w',
        format='[%(asctime)s| %(levelname)s| %(processName)s] %(message)s' # 日志格式
    )
