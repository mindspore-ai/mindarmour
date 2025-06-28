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
import mindspore as ms
from MindsporeCode.vfl.vfl import VFL
from mindspore import ops


class VILLAIN_VFL(VFL):
    """
    VILLAIN_VFL system
    """

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
            for ep in range(self.args['target_epochs']):
                loss_list = []
                self.set_train()
                self.set_current_epoch(ep)

                for batch_idx, (X, Y_batch, old_imgb, indices) in enumerate(self.train_loader):
                    # X: B,K,3,32,16 tensor
                    # import traceback
                    # print("DEBUG: backdoor_label = ", self.args['backdoor_label'])
                    # traceback.print_stack()

                    party_X_train_batch_dict = dict()
                    if self.args['n_passive_party'] < 2:
                        # X = ops.transpose(X, (1, 0, 2, 3, 4))
                        # 0627修改处理criteo
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

                # not evaluate main-task performance if evaluating execution time
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

                    # set epsilon by the var
                    if ep == self.args['backdoor_epochs']:
                        self.set_eval()
                        self.target_sample = []
                        features = []
                        for batch_idx, (X, Y_batch, old_imgb, indices) in enumerate(self.train_loader):
                            # X: B,K,3,32,16 tensor
                            target_indices = []
                            for i in range(len(indices)):
                                if Y_batch[i] == self.args['backdoor_label']:
                                    target_indices.append(i)
                            if self.args['n_passive_party'] < 2:
                                # X = ops.transpose(X, (1, 0, 2, 3, 4))
                                # 0627修改处理criteo
                                if self.args['dataset'] != 'criteo':
                                    X = ops.transpose(X, (1, 0, 2, 3, 4))
                                else:
                                    X_list = [X[:, 0, :], X[:, 1, :]]
                                    X = X_list

                                _, Xb_batch = X
                            else:
                                Xb_batch = X[:, self.adversary + 1:self.adversary + 2].squeeze(1)
                            H_features = self.party_dict[self.adversary].bottom_model.forward(Xb_batch)  # 64,10
                            for i in target_indices:
                                self.target_sample.append(H_features[i])
                            if len(features) == 0:
                                features = H_features
                            else:
                                features = ms.ops.cat((features, H_features), axis=0)  # 10000,10

                        self.target_sample = ms.ops.stack(self.target_sample)  # [500,10]

                        norms = 0
                        for target_feature in self.target_sample:
                            norms += ms.ops.norm(target_feature, 2)
                        # 0621 epoch>5报错
                        # std_norms = ms.ops.std(features, axis=0)
                        std_norms = self.safe_std(features, axis=0)
                        stride_pattern = [0] * len(features[0])
                        temp = []
                        # print(self.args['m_dimension'] // 2)
                        for i in range(self.args['m_dimension'] // 2):
                            if i % 2 == 0:
                                temp = temp + [-1, -1]
                            else:
                                temp = temp + [1, 1]
                        if len(temp) < self.args['m_dimension']:
                            h = temp[-1] * -1
                            temp.append(h)
                        sorted, indices = ms.ops.sort(std_norms, descending=True)
                        # print(stride_pattern, indices, temp)
                        for i in range(self.args['m_dimension']):
                            stride_pattern[indices[i]] = temp[i]
                        stride_pattern = ms.tensor(stride_pattern)
                        self.party_dict[self.adversary].feature_pattern = std_norms * self.args['epsilon'] * stride_pattern

    def write(self):
        # save VFL
        if self.args['save_model']:
            self.party_dict[self.adversary].save_data()
            self.save()

    @staticmethod
    def safe_std(x, axis=0, keepdims=False):
        mean = ops.mean(x, axis=axis, keep_dims=True)
        var = ops.mean((x - mean)**2, axis=axis, keep_dims=keepdims)
        std = ops.sqrt(var)
        return std
