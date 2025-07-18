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
import os
from MindsporeCode.vfl.vfl import VFL
from mindspore import ops


class Direct_VFL(VFL):
    """
    Direct attack VFL system
    """
    def train(self):
        '''
        train the vfl model
        :return:
        '''
        # load VFL
        if self.args['load_model']:
            raise ValueError('do not support load model for direct label inference attack')
        # train VFL
        else:
            for ep in range(self.args['target_epochs']):
                loss_list = []
                self.set_train()
                self.set_current_epoch(ep)
                self.party_dict[self.adversary].inferred_correct = 0
                self.party_dict[self.adversary].inferred_wrong = 0

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
                        active_X_batch = X[:, 0:1].squeeze(1)         #### 第一个特征给active party
                        for i in range(self.args['n_passive_party']):
                            party_X_train_batch_dict[i] = X[:, i+1:i+2].squeeze(1)         ####

                    # for evaluation
                    self.party_dict[self.adversary].batch_label = Y_batch
                    
                    # visualization image for web
                    if ep == 0 and batch_idx == 0:
                        output_txt_path = '../temp_output/' +  self.args['file_time'] + '-image-report.txt'
                        if not os.path.exists(output_txt_path):
                            from PIL import Image
                            import base64
                            from io import BytesIO
                            image = self.train_loader.children[0].source.data[indices[0]]

                            print("dataset: ", self.args['dataset'])
                            print("image shape: ", image.shape)
                            if self.args['dataset'] != 'bhi' and self.args['dataset'] != 'criteo':
                                image = image.transpose(1, 2, 0)

                            if self.args['dataset'] != 'criteo':
                                if len(image.shape) == 4:
                                    image = Image.fromarray(image[0])
                                else:
                                    image = Image.fromarray(image)
                            buffered = BytesIO()

                            # criteo
                            if self.args["dataset"] != "criteo":
                                image.save(buffered, format="PNG")
                                image_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                                with open(output_txt_path, "w") as txt_file:
                                    txt_file.write(image_str)
                            else:
                                with open(output_txt_path, "w") as txt_file:
                                    txt_file.write('No image data available for criteo dataset.')

                    loss, grad_list = self.fit(active_X_batch, Y_batch, party_X_train_batch_dict, indices)
                    loss_list.append(loss)
                self.scheduler_step()

                # not evaluate main-task performance if evaluating execution time
                if not self.args['time']:
                    # compute main-task accuracy
                    ave_loss = np.sum(loss_list)/len(self.train_loader.children[0])
                    self.set_state('train')
                    self.train_acc = self.predict(self.train_loader, num_classes=self.args['num_classes'],
                                       dataset=self.args['dataset'], top_k=self.top_k,
                                       n_passive_party=self.args['n_passive_party'])
                    self.set_state('test')
                    self.test_acc = self.predict(self.test_loader, num_classes=self.args['num_classes'],
                                       dataset=self.args['dataset'], top_k=self.top_k,
                                       n_passive_party=self.args['n_passive_party'])
                    self.inference_acc = self.party_dict[self.adversary].inferred_correct / \
                                         (self.party_dict[self.adversary].inferred_correct + self.party_dict[self.adversary].inferred_wrong)
                    logging.info(
                        "--- epoch: {}, train loss: {}, train_acc: {}%, test acc: {}%, direct label inference accuracy: {}".format(
                            ep, ave_loss, self.train_acc * 100, self.test_acc * 100, self.inference_acc))
                    self.record_train_acc.append(self.train_acc)
                    self.record_test_acc.append(self.test_acc)
                    self.record_loss.append(ave_loss)
                    self.record_attack_metric.append(self.inference_acc)