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
import mindspore
import logging
from MindsporeCode.common.parser import get_args
from MindsporeCode.evaluate.MainTask import Normal, Backdoor, Label_inference
mindspore.set_context(mode=mindspore.GRAPH_MODE)

def run_mindspore(input_args):

    # print("当前工作目录:", os.getcwd())
    # print("尝试加载配置文件路径:", os.path.abspath(input_args['config']))

    args = get_args(input_args['config'], input_args['file_time'])
    input_args['step_gamma'] = 0.1

    args['attack'] = input_args['attack']
    args['backdoor'] = input_args['backdoor']
    args['label_inference_attack'] = input_args['label_inference_attack']
    args['target_epochs'] = input_args['epochs']
    args['passive_bottom_lr'] = input_args['lr']
    args['active_bottom_lr'] = input_args['lr']
    args['active_top_lr'] = input_args['lr']
    args['target_batch_size'] = input_args['batch_size']
    args['passive_bottom_gamma'] = input_args['step_gamma']
    args['active_bottom_gamma'] = input_args['step_gamma']
    args['active_top_gamma'] = input_args['step_gamma']
    # args['active_top_trainable'] = input_args['top_model']
    # args['n_passive_party'] = input_args['n_party'] - 1
    args['aggregate'] = input_args['aggregate']
    args['cuda'] = input_args['cuda']
    args['file_time'] = input_args['file_time']

    # default params
    args['model_type'] = 'Resnet'  # change VGG to Resnet
    args['debug'] = False
    args['n_passive_party'] = 1
    # args['aggregate'] = 'Concate'
    args['active_top_trainable'] = True
    if args['label_inference_attack'] == 'direct_attack':
        args['active_top_trainable'] = False
    args['trigger'] = 'pixel'
    args['trigger_add'] = False

    if args['cuda']:
        # mindspore.context.set_context(device_target="GPU")
        mindspore.context.set_context(device_target="GPU", device_id=0)
        # from mindspore import Profiler
        # profiler = Profiler(output_path="profiler_data")
    else:
        mindspore.context.set_context(device_target="CPU")

    length_dict = {'cifar10':50000, 'cifar100':50000, 'cinic':180000, 'bhi':345910, 'criteo':100000}
    epochs_dict = {'cifar10':5, 'cifar100':5, 'cinic':2, 'bhi':2, 'criteo':2}

    # the index of adv in all parties
    if args['n_passive_party'] == 1:
        args['adversary'] = 1
    elif args['n_passive_party'] == 3:
        args['adversary'] = 2
    elif args['n_passive_party'] == 7:
        args['adversary'] = 5
    else:
        args['adversary'] = (args['n_passive_party'] - 1) // 2

    logging.info('frame: {}, config: {}, attack: {}, backdoor: {}, label_inference_attack: {}'.format(input_args['framework'], input_args['config'], args['attack'], args['backdoor'], args['label_inference_attack']))
    if not args['attack']:
        normal_train = Normal(args)
        normal_train.define()
        normal_train.run()
    elif args['backdoor'] != 'no':
        # if args['backdoor'] in ['lr_ba','villain']:
        #     return
        # if args['label_inference_attack'] == 'g_r':
        #     args['model_type'] = 'Resnet'

        # if args['backdoor'] == 'lr_ba':
        #     return
        if args['label_inference_attack'] in ['g_r', 'villain']:
            args['model_type'] = 'Resnet'

        args['poison_rate'] = input_args['poison_rate']
        if args['target_train_size'] == -1:
            temp = length_dict[args['dataset']]
        else:
            temp = args['target_train_size']
        args['backdoor_train_size'] = int(args['poison_rate'] * temp)
        args['backdoor_epochs'] = epochs_dict[args['dataset']]
        args['amplify_ratio'] = 2
        args['m_dimension'] = 10
        args['epsilon'] = 0.4
        # if args['label_inference_attack'] == 'g_r':
        #     args['amplify_ratio'] = 1
        if args['backdoor'] == 'villain':
            args['trigger'] = 'feature'
            args['trigger_add'] = True
        else:
            args['trigger'] = 'pixel'
            args['trigger_add'] = False

        Backdoor_attack = Backdoor(args)
        Backdoor_attack.define()
        Backdoor_attack.run()
    elif args['label_inference_attack'] != 'no':
        # if args['label_inference_attack'] in ['passive_model_completion', 'active_model_completion']:
        #     return
        if args['label_inference_attack'] == 'direct_attack':
            args['model_type'] = 'Resnet'
        Label_inference_attack = Label_inference(args)
        Label_inference_attack.define()
        Label_inference_attack.run()

    # profiler.analyse()
