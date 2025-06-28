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
import numpy as np
from MindsporeCode.common.utils import accuracy
from mindspore import ops
from datetime import datetime


class VFL(object):
    """
    VFL system
    """
    def __init__(self,train_loader, test_loader, backdoor_test_loader, party_list, args):
        super(VFL,self).__init__()
        self.active_party = party_list[0]
        self.party_dict = dict()  # passive parties dict
        self.party_ids = list()  # id list of passive parties
        self.args = args
        for index, party in enumerate(party_list[1:]):
            self.add_party(id=index, party_model=party)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.backdoor_test_loader = backdoor_test_loader
        self.top_k = 1
        if self.args['dataset'] == 'cifar100':
            self.top_k = 5
        self.state = None
        self.is_attack = False
        # the index in passive parties
        self.adversary = self.args['adversary'] - 1
        self.record_loss = []
        self.record_train_acc = []
        self.record_test_acc = []
        self.record_attack_metric = []
        self.record_results = []

    def add_party(self, *, id, party_model):
        """
        add passive party
        :param id: passive party id
        :param party_model: passive party
        """
        self.party_dict[id] = party_model
        self.party_ids.append(id)

    def set_current_epoch(self, ep):
        """
        set current train epoch
        :param ep: current train epoch
        """
        self.active_party.set_epoch(ep)
        for i in self.party_ids:
            self.party_dict[i].set_epoch(ep)


    def set_state(self, type):
        self.state = type
        if type == 'attack':
            self.is_attack = True
        else:
            self.is_attack = False

    def train(self):
        '''
        train the vfl model
        :return:
        '''
        # load VFL
        if self.args['load_model']:
            self.load()
            self.set_state('test')
            self.train_acc = self.predict(self.train_loader, num_classes=self.args['num_classes'],
                                     dataset=self.args['dataset'], top_k=self.top_k,
                                     n_passive_party=self.args['n_passive_party'])
            self.set_state('test')
            self.test_acc = self.predict(self.test_loader, num_classes=self.args['num_classes'],
                                    dataset=self.args['dataset'], top_k=self.top_k,
                                    n_passive_party=self.args['n_passive_party'])
            if self.backdoor_test_loader is not None:
                self.set_state('attack')
                self.backdoor_acc = self.predict(self.backdoor_test_loader, num_classes=self.args['num_classes'],
                                            dataset=self.args['dataset'], top_k=self.top_k,
                                            n_passive_party=self.args['n_passive_party'])
                self.set_state('test')
                print('train_acc: {}, test_acc: {}, backdoor_acc:{}'.format(self.train_acc, self.test_acc, self.backdoor_acc))
            else:
                print('train_acc: {}, test_acc: {}'.format(self.train_acc, self.test_acc))
        # train VFL
        else:
            last_time = datetime.now()
            for ep in range(self.args['target_epochs']):
                loss_list = []
                self.set_train()
                self.set_current_epoch(ep)

                for batch_idx, (X, Y_batch, old_imgb, indices) in enumerate(self.train_loader):
                    party_X_train_batch_dict = dict()
                    if self.args['n_passive_party'] < 2:
                        if self.args['dataset'] != 'criteo':
                            X = ops.transpose(X, (1, 0, 2, 3, 4))
                        else:
                            X_list = [X[:, 0, :], X[:, 1, :]]
                            X = X_list

                        active_X_batch, Xb_batch = X
                        party_X_train_batch_dict[0] = Xb_batch
                    else:
                        active_X_batch = X[:, 0:1].squeeze(1)
                        for i in range(self.args['n_passive_party']):
                            party_X_train_batch_dict[i] = X[:, i + 1:i + 2].squeeze(1)
                    loss, grad_list = self.fit(active_X_batch, Y_batch, party_X_train_batch_dict, indices)
                    loss_list.append(loss)
                self.scheduler_step()

                # not evaluate main-task performan、ce if evaluating execution time
                if not self.args['time']:
                    # compute main-task accuracy
                    ave_loss = np.sum(loss_list) / len(self.train_loader.children[0])
                    self.set_state('train')
                    self.train_acc = self.predict(self.train_loader, num_classes=self.args['num_classes'],
                                                  dataset=self.args['dataset'], top_k=self.top_k,
                                                  n_passive_party=self.args['n_passive_party'])
                    self.set_state('test')
                    self.test_acc = self.predict(self.test_loader, num_classes=self.args['num_classes'],
                                                 dataset=self.args['dataset'], top_k=self.top_k,
                                                 n_passive_party=self.args['n_passive_party'])
                    self.record_train_acc.append(self.train_acc)
                    self.record_test_acc.append(self.test_acc)
                    self.record_loss.append(ave_loss)
                    # compute backdoor task accuracy
                    if self.backdoor_test_loader is not None:
                        self.set_state('attack')
                        self.backdoor_acc = self.predict(self.backdoor_test_loader,
                                                         num_classes=self.args['num_classes'],
                                                         dataset=self.args['dataset'], top_k=self.top_k,
                                                         n_passive_party=self.args['n_passive_party'])
                        self.set_state('test')
                        logging.info(
                            "--- epoch: {}, train loss: {}, train_acc: {}%, test acc: {}%, backdoor acc: {}%".format(ep,
                                                                                                                     ave_loss,
                                                                                                                     self.train_acc * 100,
                                                                                                                     self.test_acc * 100,
                                                                                                                     self.backdoor_acc * 100))
                        self.record_attack_metric.append(self.backdoor_acc)
                    else:
                        logging.info("--- epoch: {}, train loss: {}, train_acc: {}%, test acc: {}%".format(ep, ave_loss,
                                                                                                           self.train_acc * 100,
                                                                                                           self.test_acc * 100))

    def fit(self, active_X, y, party_X_dict, indices):
        """
        VFL training in one batch
        :param active_X: features of active party
        :param y: labels
        :param dict party_X_dict: features of passive parties, the key is passive party id
        :param indices: indices of samples in current batch
        :return: loss
        """
        # set features and labels for active party
        self.active_party.set_batch(active_X, y)
        self.active_party.set_indices(indices)

        # set features for all passive parties
        for idx, party_X in party_X_dict.items():
            self.party_dict[idx].set_batch(party_X, indices)

        # all passive parties output latent representations and upload them to active party
        comp_list = []
        for id in self.party_ids:
            party = self.party_dict[id]
            logits = party.send_components()
            comp_list.append(logits)
        self.active_party.receive_components(component_list=comp_list)

        # active party compute gradients based on labels and update parameters of its bottom model and top model
        self.active_party.fit()
        loss = self.active_party.get_loss()

        # active party send gradients to passive parties, then passive parties update parameters of their bottom model
        parties_grad_list = self.active_party.send_gradients()
        grad_list = []
        for index, id in enumerate(self.party_ids):
            party = self.party_dict[id]
            grad = party.receive_gradients(parties_grad_list[index])
            grad_list.append(grad)

        return loss, grad_list

    def save(self):
        """
        save all models in VFL, including top model and all bottom models
        """
        self.active_party.save()
        for id in self.party_ids:
            self.party_dict[id].save()

    def load(self, load_attack=False):
        """
        load all models in VFL, including top model and all bottom models

        :param load_attack: invalid
        """
        self.active_party.load()
        for id in self.party_ids:
            if load_attack and id == 0:
                self.party_dict[id].load(load_attack=True)
            else:
                self.party_dict[id].load()

    def predict(self, test_loader, num_classes, dataset, top_k=1, n_passive_party=2):
        """
        compute accuracy of VFL system on test dataset
        :param test_loader: loader of test dataset
        :param num_classes: number of dataset classes
        :param dataset: dataset name
        :param top_k: top-k accuracy
        :param n_passive_party: number of passive parties
        :param is_attack: whether to compute attack accuracy
        :return: accuracy
        """
        y_predict = []
        y_true = []

        self.set_eval()

        for batch_idx, (X, targets, old_imgb, indices) in enumerate(test_loader):

            party_X_test_dict = dict()
            if self.args['n_passive_party'] < 2:
                if self.args['dataset'] != 'criteo':
                    X = ops.transpose(X, (1, 0, 2, 3, 4))
                else:
                    X_list = [X[:, 0, :], X[:, 1, :]]
                    X = X_list
                active_X_inputs, Xb_inputs = X
                party_X_test_dict[0] = Xb_inputs
            else:
                active_X_inputs = X[:, 0:1].squeeze(1)
                for i in range(n_passive_party):
                    party_X_test_dict[i] = X[:, i + 1:i + 2].squeeze(1)
            y_true += targets.tolist()


            # for ABL defense
            if self.state == 'train':
                self.active_party.y = targets
            self.active_party.indices = indices

            y_prob_preds = self.batch_predict(active_X_inputs, party_X_test_dict)
            y_predict += y_prob_preds.tolist()

        acc = accuracy(y_true, y_predict, top_k=top_k, num_classes=num_classes, dataset=dataset, is_attack=self.is_attack)

        return acc

    def write(self):
        '''

        Returns:

        '''
        # save VFL
        if self.args['save_model']:
            self.save()

    def batch_predict(self, active_X, party_X_dict):
        """
        predict label with help of all parties
        :param active_X: features of active party
        :param dict party_X_dict: features of passive parties, the key is passive party id
        :param attack_output: latent represent output by the attacker if provided, otherwise the attacker output using bottom model
        :param is_attack: attack or not in the predict process, sr_ba is True, else False
        :return: prediction label
        """
        comp_list = []
        # passive parties send latent representations
        for id in self.party_ids:
            comp_list.append(self.party_dict[id].predict(party_X_dict[id], self.is_attack))

        # active party make the final prediction
        return self.active_party.predict(active_X, component_list=comp_list, type=self.state)

    def set_train(self):
        """
        set train mode for all parties
        """
        self.active_party.set_train()
        for id in self.party_ids:
            self.party_dict[id].set_train()

    def set_eval(self):
        """
        set eval mode for all parties
        """
        self.active_party.set_eval()
        for id in self.party_ids:
            self.party_dict[id].set_eval()

    def scheduler_step(self):
        """
        adjust learning rate for all parties during training
        """
        self.active_party.scheduler_step()
        for id in self.party_ids:
            self.party_dict[id].scheduler_step()

    def zero_grad(self):
        """
        clear gradients for all parties
        """
        self.active_party.zero_grad()
        for id in self.party_ids:
            self.party_dict[id].zero_grad()
