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
from mindspore import ops

def backdoor_image_predict(self, test_loader, num_classes, dataset, top_k=1, n_passive_party=2):
    y_backdoor_dict = {}
    image = None
    image_indices = []

    self.set_eval()

    self.set_state('attack')
    for batch_idx, (X, targets, old_imgb, indices) in enumerate(test_loader):
        party_X_test_dict = dict()
        if self.args['n_passive_party'] < 2:
            # X = ops.transpose(X, (1, 0, 2, 3, 4))
            # 0627修改适配criteo
            if self.args['dataset'] != 'criteo':
                X = ops.transpose(X, (1, 0, 2, 3, 4))
            else:
                X_list = [X[:, 0, :], X[:, 1, :]]
                X = X_list
            active_X_inputs, Xb_inputs = X
            # if self.args['cuda']:
            #     active_X_inputs = active_X_inputs.cuda()
            #     Xb_inputs = Xb_inputs.cuda()
            #     targets = targets.cuda()
            #     indices = indices.cuda()
            party_X_test_dict[0] = Xb_inputs
        else:
            # if self.args['cuda']:
            #     X = X.cuda()
            #     targets = targets.cuda()
            #     indices = indices.cuda()
            active_X_inputs = X[:, 0:1].squeeze(1)
            for i in range(n_passive_party):
                party_X_test_dict[i] = X[:, i + 1:i + 2].squeeze(1)

        # for ABL defense
        if self.state == 'train':
            self.active_party.y = targets
        self.active_party.indices = indices

        each_party_X_test_dict = party_X_test_dict.copy()
        for index in range(len(indices)):
            each_active_X_inputs = ops.unsqueeze(active_X_inputs[index], dim=0)
            for i in range(n_passive_party):
                each_party_X_test_dict[i] = ops.unsqueeze(party_X_test_dict[i][index], dim=0)
            # y_true = targets.data.tolist()[index]
            y_backdoor = self.batch_predict(each_active_X_inputs, each_party_X_test_dict)[0]
            predicted_class = ops.argmax(y_backdoor)
            if predicted_class == self.args['backdoor_label']:
                image_indices.append(indices[index])
                y_backdoor_dict[indices[index].item()] = y_backdoor

    self.set_state('test')
    old_indice_map = test_loader.children[0].source.indice_map
    old_backdoor_indices = test_loader.children[0].source.backdoor_indices
    test_loader.children[0].source.indice_map = None
    test_loader.children[0].source.backdoor_indices = None
    for batch_idx, (X, targets, old_imgb, indices) in enumerate(test_loader):
        party_X_test_dict = dict()
        if self.args['n_passive_party'] < 2:
            # X = ops.transpose(X, (1, 0, 2, 3, 4))
            # 0627修改适配criteo
            if self.args['dataset'] != 'criteo':
                X = ops.transpose(X, (1, 0, 2, 3, 4))
            else:
                X_list = [X[:, 0, :], X[:, 1, :]]
                X = X_list
            active_X_inputs, Xb_inputs = X
            # if self.args['cuda']:
            #     active_X_inputs = active_X_inputs.cuda()
            #     Xb_inputs = Xb_inputs.cuda()
            #     targets = targets.cuda()
            #     indices = indices.cuda()
            party_X_test_dict[0] = Xb_inputs
        else:
            # if self.args['cuda']:
            #     X = X.cuda()
            #     targets = targets.cuda()
            #     indices = indices.cuda()
            active_X_inputs = X[:, 0:1].squeeze(1)
            for i in range(n_passive_party):
                party_X_test_dict[i] = X[:, i + 1:i + 2].squeeze(1)

        # for ABL defense
        if self.state == 'train':
            self.active_party.y = targets
        self.active_party.indices = indices

        image_str = None
        each_party_X_test_dict = party_X_test_dict.copy()
        for indice in indices:
            if indice in image_indices:
                # print("case 1")
                index = ops.nonzero(indices == indice)[0][0]
            #0523问题修复
            elif indice == indices[-1]:
                # print("case 2")
                index = ops.nonzero(indices == indice)[0][0]
            else:
                # print("case 3")
                continue

            each_active_X_inputs = ops.unsqueeze(active_X_inputs[index], dim=0)
            for i in range(n_passive_party):
                each_party_X_test_dict[i] = ops.unsqueeze(party_X_test_dict[i][index], dim=0)

            y_clean = self.batch_predict(each_active_X_inputs, each_party_X_test_dict)[0]

            predicted_class = ops.argmax(y_clean)
            if predicted_class != self.args['backdoor_label'] or indice == indices[-1]:
                # image_indices = indice
                # 0523问题修复 待测试,后面用的参数也从image_indices改为select_indice
                selected_indice = indice
                if selected_indice.item() not in y_backdoor_dict:
                    # print(f"Warning: selected indice {selected_indice.item()} not in y_backdoor_dict, skip.")
                    continue

                y_backdoor = y_backdoor_dict[selected_indice.item()]
                from PIL import Image
                import base64
                from io import BytesIO
                image = test_loader.children[0].source.data[selected_indice]

                # 0627 加criteo重改了下面这一大段
                # image = image.transpose(1, 2, 0) # 32,32,3
                # # image = Image.fromarray(image)
                # # print(f"ms image report------image shape：{image.shape}") # 32,32,3
                # # 修改格式 for bhi
                # if len(image.shape) == 4 and self.args["dataset"] != "criteo":
                #     image = Image.fromarray(image[0])
                # else:
                #     image = Image.fromarray(image)
                if self.args["dataset"] != "criteo":
                    # 修改格式 for bhi
                    if len(image.shape) == 4:
                        image = Image.fromarray(image[0])
                    else:
                        image = image.transpose(1, 2, 0)  # 32,32,3
                        image = Image.fromarray(image)

                buffered = BytesIO()
                # image.save(buffered, format="PNG")
                # image_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                # 0627criteo加的判断
                if self.args["dataset"] != "criteo":
                    image.save(buffered, format="PNG")
                    image_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                break

        if image is not None:
            break

    return image_str, y_clean, y_backdoor

def visualization_with_images(self, ema_top_model=None):
    if self.args['attack'] and self.args['backdoor'] != 'no':
        # return images and other info
        self.set_state('attack')
        image_str, y_clean, y_backdoor = backdoor_image_predict(self, self.backdoor_test_loader,
                                                                num_classes=self.args['num_classes'],
                                                                dataset=self.args['dataset'], top_k=self.top_k,
                                                                n_passive_party=self.args['n_passive_party'])
        return image_str, y_clean, y_backdoor
    elif self.args['attack'] and 'model_completion' in self.args['label_inference_attack']:
        from methods.model_completion.model_completion import predict
        image_str, y_predict, y_infer = predict(self, self.train_loader, ema_top_model)
        return image_str, y_predict, y_infer
    else:
        return None
    
def append_predictions_to_file(file_path, clean_pred, attack_pred):
    # 准备写入的字符串内容
    # clean_str = ", ".join(map(str, clean_pred))  # 将clean预测列表转成字符串
    # attack_str = ", ".join(map(str, attack_pred))  # 将attack预测列表转成字符串
    clean_str = ", ".join([str(p.item()) for p in clean_pred])  # 将 clean 预测列表的 tensor 转换为数值
    attack_str = ", ".join([str(p.item()) for p in attack_pred])  # 将 attack 预测列表的 tensor 转换为数值

    # 需要写入的格式化字符串
    data_to_append = f"clean prediction: [{clean_str}], attack prediction: [{attack_str}],"

    # 打开文件并追加内容
    with open(file_path, "a") as file:
        # 先写一个空行，再写入数据
        file.write("\n\n")  # 添加空行
        file.write(data_to_append + "\n")  # 追加数据并换行

    print(f"数据已追加到 {file_path}")

def append_int_to_file(file_path, clean_int, attack_int):
    # 准备写入的字符串内容，将单个整数转换为字符串
    clean_str = str(clean_int)  # 将 clean 整数转换为字符串
    attack_str = str(attack_int)  # 将 attack 整数转换为字符串

    # 需要写入的格式化字符串
    data_to_append = f"clean prediction: [{clean_str}], attack prediction: [{attack_str}],"

    # 打开文件并追加内容
    with open(file_path, "a") as file:
        # 先写一个空行，再写入数据
        file.write("\n\n")  # 添加空行
        file.write(data_to_append + "\n")  # 追加数据并换行

    print(f"整数数据已追加到 {file_path}")