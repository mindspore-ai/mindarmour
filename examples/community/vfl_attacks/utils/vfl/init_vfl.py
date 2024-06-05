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
Initialization of Vertical Federated Learning (VFL) participants, including models and participant types.

This module handles the initialization of participants in a Vertical Federated Learning (VFL) setup,
defining the models used by each participant and specifying the type of each participant.
"""
from mindspore import ops
from party.active_party import VFLActiveModel
from party.passive_party import VFLPassiveModel
from methods.g_r.g_r_passive_party import GRPassiveModel
from methods.direct_attack.direct_attack_passive_party import DirectAttackPassiveModel

__all__ = ["Init_Vfl"]

class Init_Vfl(object):
    """
    Initialize the vfl.
    """
    def __init__(self, args):
        super(Init_Vfl, self).__init__()
        self.args = args

    def get_vfl(self, bottoms, active, train_dl, backdoor_target_indices=None, backdoor_indices=None):
        """
        Generate the VFL system and set the parties.

        Args:
            args (dict): Configuration for the VFL system.
            backdoor_indices (List[int]): Indices of backdoor samples in the normal train dataset,
                                        used for gradient replacement.
            backdoor_target_indices (List[int]): Indices of samples labeled as the backdoor
                                        class in the normal train dataset.
        """
        self.traindl = train_dl
        self.backdoor_target_indices = backdoor_target_indices
        self.backdoor_indices = backdoor_indices

        active_bottom_model = bottoms[0]
        party_model_list = []
        for i in range(0, self.args['n_passive_party']):
            passive_party_model = bottoms[i+1]
            party_model_list.append(passive_party_model)

        active_top_model = None
        if self.args['active_top_trainable']:
            active_top_model = active

        active_party = VFLActiveModel(bottom_model=active_bottom_model,
                                      args=self.args,
                                      top_model=active_top_model)

        self.party_list = [active_party]
        for i, model in enumerate(party_model_list):
            if self.args['backdoor'] == 'g_r' and i == self.args['adversary'] - 1:
                passive_party = GRPassiveModel(bottom_model=model,
                                               amplify_ratio=self.args['amplify_ratio'], id=i, args=self.args)
                backdoor_X = {}
                if self.traindl is not None:
                    for X, _, _, indices in self.traindl:
                        temp_indices = list(set(self.backdoor_indices) & set(indices.tolist()))
                        if len(temp_indices) > 0:
                            if self.args['n_passive_party'] < 2:
                                X = ops.transpose(X, (1, 0, 2, 3, 4))
                                _, Xb_batch = X
                            else:
                                Xb_batch = X[:, self.args['adversary']:self.args['adversary']+1].squeeze(1)
                            for temp in temp_indices:
                                backdoor_X[temp] = Xb_batch[indices.tolist().index(temp)]
                passive_party.set_backdoor_indices(self.backdoor_target_indices, self.backdoor_indices, backdoor_X)
            elif self.args['label_inference_attack'] == 'direct_attack' and i == self.args['adversary'] - 1:
                passive_party = DirectAttackPassiveModel(bottom_model=model, id=i, args=self.args)
            else:
                passive_party = VFLPassiveModel(bottom_model=model, id=i, args=self.args)
            self.party_list.append(passive_party)
