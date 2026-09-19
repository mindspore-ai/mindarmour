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
from model.init_passive_model import init_bottom_model
from MindsporeCode.model.init_active_model import init_top_model
from MindsporeCode.party.active_party import VFLActiveModel
from MindsporeCode.party.passive_party import VFLPassiveModel
from MindsporeCode.datasets.base_dataset import get_dataloader
from MindsporeCode.methods.villain.villain_passive_party import Villain_PassiveModel
from MindsporeCode.methods.g_r.g_r_passive_party import GRPassiveModel
from MindsporeCode.methods.direct_attack.direct_attack_passive_party import Direct_Attack_PassiveModel
import logging
from mindspore import ops

class Init_vfl(object):
    def __init__(self, args):
        super(Init_vfl, self).__init__()
        self.args = args
    def load_data(self):
        logging.info("################################ Prepare Data ############################")
        if not self.args['attack']:
            self.train_loader, self.test_loader, self.backdoor_test_loader = get_dataloader(self.args)[:3]
        elif self.args['backdoor'] == 'lr_ba':
            self.train_loader, self.test_loader, self.backdoor_test_loader, self.backdoor_train_loader, _, self.backdoor_indices, _, self.labeled_loader, self.unlabeled_loader = get_dataloader(self.args)[0:9]
        elif self.args['backdoor'] == 'villain':
            self.train_loader, self.test_loader, self.backdoor_test_loader, _, _, _, \
            self.backdoor_target_indices, _, _, self.villain_train_dl = get_dataloader(self.args)
        elif self.args['backdoor'] == 'g_r':
            self.train_loader, self.test_loader, self.backdoor_test_loader, _, self.g_r_train_dl, self.backdoor_indices, self.backdoor_target_indices = get_dataloader(self.args)[0:7]
        elif self.args['label_inference_attack']:
            self.train_loader, self.test_loader, _, _, _, _, _, self.labeled_loader, self.unlabeled_loader = get_dataloader(self.args)[0:9]
        else:
            self.train_loader, self.test_loader, self.backdoor_test_loader = get_dataloader(self.args)[:3]


    def get_vfl(self):
        """
        generate VFL system, set the parties
        :param args: configuration
        :param backdoor_indices: indices of backdoor samples in normal train dataset, used by gradient-replacement
        :param backdoor_target_indices: indices of samples labeled backdoor class in normal train dataset, used by gradient-replacement
        :return: VFL system
        """
        # build bottom model for active party
        active_bottom_model = init_bottom_model('active', self.args)

        # build bottom model for passive parties
        party_model_list = list()
        for i in range(0, self.args['n_passive_party']):
            passive_party_model = init_bottom_model('passive', self.args)
            party_model_list.append(passive_party_model)

        # build top model for active party
        active_top_model = None
        if self.args['active_top_trainable']:
            active_top_model = init_top_model(self.args)

        # generate active party
        active_party = VFLActiveModel(bottom_model=active_bottom_model,
                                      args=self.args,
                                      top_model=active_top_model)

        # generate passive parties
        self.party_list = [active_party]
        for i, model in enumerate(party_model_list):
            if self.args['backdoor'] == 'g_r' and i == (self.args['adversary'] - 1):
                passive_party = GRPassiveModel(bottom_model=model,
                                               amplify_ratio=self.args['amplify_ratio'], id=i, args=self.args)
                backdoor_X = dict()
                if self.g_r_train_dl is not None:
                    for X, _, _, indices in self.g_r_train_dl:
                        temp_indices = list(set(self.backdoor_indices) & set(indices.tolist()))
                        if len(temp_indices) > 0:
                            if self.args['n_passive_party'] < 2:
                                if self.args['dataset'] != 'criteo':
                                    X = ops.transpose(X, (1, 0, 2, 3, 4))
                                else:
                                    X_list = [X[:, 0, :], X[:, 1, :]]
                                    X = X_list
                                _, Xb_batch = X
                            else:
                                Xb_batch = X[:, self.args['adversary']:self.args['adversary']+1].squeeze(1)
                            for temp in temp_indices:
                                backdoor_X[temp] = Xb_batch[indices.tolist().index(temp)]
                passive_party.set_backdoor_indices(self.backdoor_target_indices, self.backdoor_indices, backdoor_X)
            elif self.args['backdoor'] == 'villain' and i == (self.args['adversary'] - 1):
                passive_party = Villain_PassiveModel(bottom_model=model, amplify_ratio=self.args['amplify_ratio'], args=self.args, id=i)
                passive_party.set_backdoor_indices(target_indices=self.backdoor_target_indices)
            elif self.args['label_inference_attack'] == 'direct_attack' and i == (self.args['adversary'] - 1):
                passive_party = Direct_Attack_PassiveModel(bottom_model=model, id=i, args=self.args)
            else:
                passive_party = VFLPassiveModel(bottom_model=model, id=i, args=self.args)
            self.party_list.append(passive_party)
