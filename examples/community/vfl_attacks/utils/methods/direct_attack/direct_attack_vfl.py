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
Vertical Federated Learning (VFL) for direct label inference attacks.
"""
import numpy as np
from mindspore import ops
from vfl.vfl import VFL

__all__ = ["DirectVFL"]

class DirectVFL(VFL):
    """
    VFL for direct label inference attack.
    """
    def train(self):
        """
        Load or train the vfl models.
        """
        if self.args['load_model']:
            raise ValueError('do not support load model for direct label inference attack')
        else:
            for ep in range(self.args['target_epochs']):
                loss_list = []
                self.set_train()
                self.set_current_epoch(ep)
                self.party_dict[self.adversary].inferred_correct = 0
                self.party_dict[self.adversary].inferred_wrong = 0

                for _, (X, Y_batch, _, indices) in enumerate(self.train_loader):
                    party_X_train_batch_dict = dict()
                    if self.args['n_passive_party'] < 2:
                        X = ops.transpose(X, (1, 0, 2, 3, 4))
                        active_X_batch, Xb_batch = X
                        party_X_train_batch_dict[0] = Xb_batch
                    else:
                        active_X_batch = X[:, 0:1].squeeze(1)
                        for i in range(self.args['n_passive_party']):
                            party_X_train_batch_dict[i] = X[:, i+1:i+2].squeeze(1)

                    self.party_dict[self.adversary].batch_label = Y_batch

                    loss, _ = self.fit(active_X_batch, Y_batch, party_X_train_batch_dict, indices)
                    loss_list.append(loss)
                self.scheduler_step()

                # Compute main-task accuracy.
                ave_loss = np.sum(loss_list)/len(self.train_loader.children[0])
                self.set_state('train')
                self.train_acc = self.predict(self.train_loader, num_classes=self.args['num_classes'],top_k=self.top_k,
                                              n_passive_party=self.args['n_passive_party'])
                self.set_state('test')
                self.test_acc = self.predict(self.test_loader, num_classes=self.args['num_classes'],top_k=self.top_k,
                                             n_passive_party=self.args['n_passive_party'])
                self.inference_acc = self.party_dict[self.adversary].inferred_correct / \
                                     (self.party_dict[self.adversary].inferred_correct +
                                      self.party_dict[self.adversary].inferred_wrong)
                print(f"--- epoch: {ep}, train loss: {ave_loss}, train_acc: {self.train_acc * 100}%, "
                      f"test acc: {self.test_acc * 100}%, direct label inference accuracy: {self.inference_acc}")
                self.record_train_acc.append(self.train_acc)
                self.record_test_acc.append(self.test_acc)
                self.record_loss.append(ave_loss)
                self.record_attack_metric.append(self.inference_acc)
