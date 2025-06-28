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
import logging
from MindsporeCode.vfl.init_vfl import Init_vfl
from MindsporeCode.vfl.vfl import VFL
from methods.villain.villain_vfl import VILLAIN_VFL
from MindsporeCode.methods.lr_ba.lr_ba_vfl import Lr_ba_VFL
from methods.model_completion.model_completion import passive_model_completion
from MindsporeCode.methods.direct_attack.direct_attack_vfl import Direct_VFL
from MindsporeCode.common.image_report import visualization_with_images, append_predictions_to_file

class Normal(object):
    def __init__(self, args):
        self.args = args

    def define(self):
        init_vfl = Init_vfl(self.args)
        init_vfl.load_data()
        init_vfl.get_vfl()
        self.VFL_framework = VFL(init_vfl.train_loader, init_vfl.test_loader, init_vfl.backdoor_test_loader,
                            init_vfl.party_list, self.args)
    def run(self):
        self.VFL_framework.train()
        if not self.args['load_time']:
            self.VFL_framework.record_results = [self.VFL_framework.record_loss[-1], self.VFL_framework.train_acc, self.VFL_framework.test_acc, self.VFL_framework.backdoor_acc]
            self.VFL_framework.write()
            logging.info('self.VFL_framework.record_results: {}'.format(self.VFL_framework.record_results))
        else:
            self.VFL_framework.record_results = [self.VFL_framework.train_acc, self.VFL_framework.test_acc]


class Backdoor(object):
    def __init__(self, args):
        self.args = args

    def define(self):
        self.init_vfl = Init_vfl(self.args)
        self.init_vfl.load_data()
        self.init_vfl.get_vfl()
        if self.args['backdoor'] == 'villain':
            self.VFL_framework = VILLAIN_VFL(self.init_vfl.villain_train_dl, self.init_vfl.test_loader, self.init_vfl.backdoor_test_loader, self.init_vfl.party_list, self.args)
        elif self.args['backdoor'] == 'lr_ba':
            self.VFL_framework = Lr_ba_VFL(self.init_vfl.train_loader, self.init_vfl.test_loader, self.init_vfl.backdoor_test_loader, self.init_vfl.party_list, self.args)
        elif self.args['backdoor'] == 'g_r':
            self.VFL_framework = VFL(self.init_vfl.g_r_train_dl, self.init_vfl.test_loader, self.init_vfl.backdoor_test_loader,self.init_vfl.party_list, self.args)
        else:
            self.VFL_framework = VFL(self.init_vfl.train_loader, self.init_vfl.test_loader, self.init_vfl.backdoor_test_loader,self.init_vfl.party_list, self.args)

    def run(self):
        self.VFL_framework.train()
        if self.args['backdoor'] != 'lr_ba' and not self.args['load_time']:
            self.VFL_framework.write()
        if not self.args['load_time']:
            self.VFL_framework.record_results = [self.VFL_framework.record_loss[-1], self.VFL_framework.train_acc, self.VFL_framework.test_acc, self.VFL_framework.backdoor_acc]
            logging.info('self.VFL_framework.record_results: {}'.format(self.VFL_framework.record_results))

        # visualization image
        result = visualization_with_images(self.VFL_framework)
        # if result is not None:
        if result[0] is not None:
            image_str, y_clean, y_backdoor = result
            output_txt_path = '../temp_output/' +  self.args['file_time'] + '-image-report.txt'
            with open(output_txt_path, "w") as txt_file:
                txt_file.write(image_str)
            append_predictions_to_file(output_txt_path, y_clean, y_backdoor)
        elif self.args['dataset'] == 'criteo':
            # for criteo dataset, we do not have image
            image_str, y_clean, y_backdoor = result
            output_txt_path = '../temp_output/' +  self.args['file_time'] + '-image-report.txt'
            with open(output_txt_path, "w") as txt_file:
                txt_file.write('No image data available for criteo dataset.')
            append_predictions_to_file(output_txt_path, y_clean, y_backdoor)
        return

class Label_inference(object):
    def __init__(self, args):
        self.args = args
        if self.args['label_inference_attack'] == 'active_model_completion':
            self.args['mal_optim'] = True

    def define(self):
        self.init_vfl = Init_vfl(self.args)
        self.init_vfl.load_data()
        self.init_vfl.get_vfl()
        if self.args['label_inference_attack'] == 'direct_attack':
            if self.args['active_top_trainable']:
                raise ValueError('direct attack does not support a trainable top model!')
            self.VFL_framework = Direct_VFL(self.init_vfl.train_loader, self.init_vfl.test_loader, None,
                                     self.init_vfl.party_list, self.args)
        else:
            self.VFL_framework = VFL(self.init_vfl.train_loader, self.init_vfl.test_loader, None,
                                 self.init_vfl.party_list, self.args)

    def run(self):
        self.VFL_framework.train()
        if 'model_completion' not in self.args['label_inference_attack']:
            self.VFL_framework.record_results = [self.VFL_framework.record_loss[-1], self.VFL_framework.train_acc, self.VFL_framework.test_acc, self.VFL_framework.inference_acc]
            logging.info('self.VFL_framework.record_results: {}'.format(self.VFL_framework.record_results))
        if 'model_completion' in self.args['label_inference_attack']:
            record_infer_train_acc, record_infer_test_acc = passive_model_completion(self.args, self.init_vfl, self.VFL_framework)
            logging.info('infer acc, train: {}, test: {}'.format(record_infer_train_acc, record_infer_test_acc))
        if not self.args['load_time']:
            self.VFL_framework.write()
