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
This module defines the processes of data distribution, training, and prediction in VFL.
"""
import numpy as np
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from mindspore import ops

__all__ = ["VFL"]

class VFL(object):
    """
    VFL system.
    """
    def __init__(self,train_loader, test_loader, init_vfl, backdoor_test_loader=None):
        super(VFL,self).__init__()
        self.active_party = init_vfl.party_list[0]
        self.party_dict = {}
        self.party_ids = []
        self.args = init_vfl.args
        for index, party in enumerate(init_vfl.party_list[1:]):
            self.add_party(id=index, party_model=party)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.backdoor_test_loader = backdoor_test_loader
        self.top_k = self.args['topk']
        self.state = None
        self.is_attack = False
        self.adversary = self.args['adversary'] - 1
        self.record_loss = []
        self.record_train_acc = []
        self.record_test_acc = []
        self.record_attack_metric = []
        self.record_results = []

    def add_party(self, *, id, party_model):
        """
        Add a passive party to the VFL system.

        Args:
            id (int): The identifier for the passive party.
            party_model (VFLPassiveModel): The passive party model to be added.
        """
        self.party_dict[id] = party_model
        self.party_ids.append(id)

    def set_current_epoch(self, ep):
        """
        Set the current train epoch.

        Args:
            ep (int): The current train epoch.
        """
        self.active_party.set_epoch(ep)
        for i in self.party_ids:
            self.party_dict[i].set_epoch(ep)


    def set_state(self, type):
        """
        Set the state of the system.

        Args:
            type (str): The state type, which can be 'train', 'test', or 'attack'.
        """
        self.state = type
        if type == 'attack':
            self.is_attack = True
        else:
            self.is_attack = False

    def train(self):
        """
        Load or train the vfl model.
        """
        if self.args['load_model']:
            self.load()
            self.set_state('test')
            self.train_acc = self.predict(self.train_loader, num_classes=self.args['num_classes'],top_k=self.top_k,
                                     n_passive_party=self.args['n_passive_party'])
            self.set_state('test')
            self.test_acc = self.predict(self.test_loader, num_classes=self.args['num_classes'],top_k=self.top_k,
                                    n_passive_party=self.args['n_passive_party'])
            if self.backdoor_test_loader is not None:
                self.set_state('attack')
                self.backdoor_acc = self.predict(self.backdoor_test_loader, num_classes=self.args['num_classes'],
                                                 top_k=self.top_k, n_passive_party=self.args['n_passive_party'])
                self.set_state('test')
                print(f'train_acc: {self.train_acc}, test_acc: {self.test_acc}, backdoor_acc:{self.backdoor_acc}')
            else:
                print(f'train_acc: {self.train_acc}, test_acc: {self.test_acc}')
        else:
            for ep in range(self.args['target_epochs']):
                loss_list = []
                self.set_train()
                self.set_current_epoch(ep)

                for batch_idx, (X, Y_batch, old_imgb, indices) in enumerate(self.train_loader):
                    party_X_train_batch_dict = {}
                    if self.args['n_passive_party'] < 2:
                        X = ops.transpose(X, (1, 0, 2, 3, 4))
                        active_X_batch, Xb_batch = X
                        party_X_train_batch_dict[0] = Xb_batch
                    else:
                        active_X_batch = X[:, 0:1].squeeze(1)
                        for i in range(self.args['n_passive_party']):
                            party_X_train_batch_dict[i] = X[:, i + 1:i + 2].squeeze(1)
                    loss, grad_list = self.fit(active_X_batch, Y_batch, party_X_train_batch_dict, indices)
                    loss_list.append(loss)
                self.scheduler_step()

                # Compute main-task accuracy.
                ave_loss = np.sum(loss_list) / len(self.train_loader.children[0])
                self.set_state('train')
                self.train_acc = self.predict(self.train_loader, num_classes=self.args['num_classes'],top_k=self.top_k,
                                              n_passive_party=self.args['n_passive_party'])
                self.set_state('test')
                self.test_acc = self.predict(self.test_loader, num_classes=self.args['num_classes'],top_k=self.top_k,
                                             n_passive_party=self.args['n_passive_party'])
                self.record_train_acc.append(self.train_acc)
                self.record_test_acc.append(self.test_acc)
                self.record_loss.append(ave_loss)
                # Compute backdoor task accuracy.
                if self.backdoor_test_loader is not None:
                    self.set_state('attack')
                    self.backdoor_acc = self.predict(self.backdoor_test_loader,
                                                     num_classes=self.args['num_classes'], top_k=self.top_k,
                                                     n_passive_party=self.args['n_passive_party'])
                    self.set_state('test')
                    print(
                        f"--- epoch: {ep}, train loss: {ave_loss}, train_acc: {self.train_acc * 100}%, "
                        f"test acc: {self.test_acc * 100}%, backdoor acc: {self.backdoor_acc * 100}%")
                    self.record_attack_metric.append(self.backdoor_acc)
                else:
                    print(
                        f"--- epoch: {ep}, train loss: {ave_loss}, train_acc: {self.train_acc * 100}%, "
                        f"test acc: {self.test_acc * 100}%")

    def fit(self, active_X, y, party_X_dict, indices):
        """
        Perform VFL training for one batch.

        Args:
            active_X (Tensor): Features of the active party.
            y (Tensor): Labels for the current batch.
            party_X_dict (dict): Features of passive parties, with party IDs as keys.
            indices (List[int]): Indices of samples in the current batch.

        Returns:
            tuple: A tuple containing the loss computed for the batch and the gradients returned by the bottom models.
                    The gradients are marked as invalid in normal training.
        """
        # Set features and labels for active party.
        self.active_party.set_batch(active_X, y)
        self.active_party.set_indices(indices)

        # Set features for all passive parties.
        for idx, party_X in party_X_dict.items():
            self.party_dict[idx].set_batch(party_X, indices)

        # All passive parties output latent representations and upload them to active party.
        comp_list = []
        for id in self.party_ids:
            party = self.party_dict[id]
            logits = party.send_components()
            comp_list.append(logits)
        self.active_party.receive_components(component_list=comp_list)

        # Active party compute gradients based on labels and update parameters of its bottom model and top model.
        self.active_party.fit()
        loss = self.active_party.get_loss()

        # Active party send gradients to passive parties, then passive parties update their bottom models.
        parties_grad_list = self.active_party.send_gradients()
        grad_list = []
        for index, id in enumerate(self.party_ids):
            party = self.party_dict[id]
            grad = party.receive_gradients(parties_grad_list[index])
            grad_list.append(grad)

        return loss, grad_list

    def save(self):
        """
        Save all models in VFL, including top model and all bottom models.
        """
        self.active_party.save()
        for id in self.party_ids:
            self.party_dict[id].save()

    def load(self, load_attack=False):
        """
        Load all models in the VFL system, including the top model and all bottom models.

        Args:
            load_attack (bool): invalid.
        """
        self.active_party.load()
        for id in self.party_ids:
            if load_attack and id == 0:
                self.party_dict[id].load(load_attack=True)
            else:
                self.party_dict[id].load()

    def predict(self, test_loader, num_classes, top_k=1, n_passive_party=2):
        """
        Compute the accuracy of the VFL system on the test dataset.

        Args:
            test_loader (DataLoader): Loader for the test dataset.
            num_classes (int): Number of classes in the dataset.
            dataset (str): Name of the dataset.
            top_k (int): Top-k value for accuracy computation.
            n_passive_party (int): Number of passive parties in the VFL system.
            is_attack (bool): Whether to compute attack accuracy.

        Returns:
            float: The computed accuracy of the VFL system.
        """
        y_predict = []
        y_true = []

        self.set_eval()

        for batch_idx, (X, targets, old_imgb, indices) in enumerate(test_loader):
            party_X_test_dict = {}
            if self.args['n_passive_party'] < 2:
                X = ops.transpose(X, (1, 0, 2, 3, 4))
                active_X_inputs, Xb_inputs = X
                party_X_test_dict[0] = Xb_inputs
            else:
                active_X_inputs = X[:, 0:1].squeeze(1)
                for i in range(n_passive_party):
                    party_X_test_dict[i] = X[:, i + 1:i + 2].squeeze(1)
            y_true += targets.tolist()

            self.active_party.indices = indices

            y_prob_preds = self.batch_predict(active_X_inputs, party_X_test_dict)
            y_predict += y_prob_preds.tolist()

        acc = self.accuracy(y_true, y_predict, top_k=top_k, num_classes=num_classes)
        return acc

    def write(self):
        '''
        Save the models.
        '''
        if self.args['save_model']:
            self.save()

    def batch_predict(self, active_X, party_X_dict):
        """
        Predict labels with help of all parties.

        Args:
            active_X (Tensor): Features of the active party.
            party_X_dict (dict): Features of passive parties, with party IDs as keys.
            attack_output (Tensor, optional): Latent representation output by the attacker if provided.
            is_attack (bool): Whether the prediction process is for an attack scenario.

        Returns:
            Tensor: The prediction labels.
        """
        comp_list = []
        # Passive parties send latent representations
        for id in self.party_ids:
            comp_list.append(self.party_dict[id].predict(party_X_dict[id], self.is_attack))

        # Active party make the final prediction
        return self.active_party.predict(active_X, component_list=comp_list, type=self.state)

    def set_train(self):
        """
        Set train mode for all parties.
        """
        self.active_party.set_train()
        for id in self.party_ids:
            self.party_dict[id].set_train()

    def set_eval(self):
        """
        Set eval mode for all parties.
        """
        self.active_party.set_eval()
        for id in self.party_ids:
            self.party_dict[id].set_eval()

    def scheduler_step(self):
        """
        Adjust learning rate for all parties during training.
        """
        self.active_party.scheduler_step()
        for id in self.party_ids:
            self.party_dict[id].scheduler_step()

    def zero_grad(self):
        """
        Clear gradients for all parties.
        """
        self.active_party.zero_grad()
        for id in self.party_ids:
            self.party_dict[id].zero_grad()

    def accuracy(self, y_true, y_pred, num_classes=None, top_k=1):
        """
        Compute model accuracy.

        Args:
            y_true (list): List of ground-truth labels.
            y_pred (list): List of prediction labels.
            dataset (str): Name of the dataset.
            num_classes (int): Number of classes in the dataset.
            top_k (int, optional): Top-k value for accuracy computation. Default is 1.
            is_attack (bool, optional): Whether to compute accuracy for attack scenarios.

        Returns:
            float: The computed model accuracy.
        """
        y_pred = np.array(y_pred)
        if np.any(np.isnan(y_pred)) or not np.all(np.isfinite(y_pred)):
            raise ValueError('accuracy y_pred is isnan')
        temp_y_pred = []
        if top_k == 1:
            for pred in y_pred:
                temp = np.max(pred)
                temp_y_pred.append(np.where(pred == temp)[0][0])
            acc = accuracy_score(y_true, temp_y_pred)
        else:
            acc = top_k_accuracy_score(y_true, y_pred, k=top_k, labels=np.arange(num_classes))
        return acc
