# Copyright 2023 Huawei Technologies Co., Ltd
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

import os
import random
import shutil


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == "__main__":

    random.seed(1)

    # 数据集文件夹路径
    DATASET_DIR = "./car_dataset"
    # 分割后文件保存文件夹路径
    SPLIT_DIR = "./car_dataset"
    # 训练集
    train_dir = os.path.join(SPLIT_DIR, "train")
    # 验证集
    valid_dir = os.path.join(SPLIT_DIR, "valid")
    # 测试集
    test_dir = os.path.join(SPLIT_DIR, "test")

    # 训练集占比
    TRAIN_PCT = 0.8
    # 验证集占比
    VALID_PCT = 0.0
    # 测试集占比
    TEST_PCT = 0.2

    for root, dirs, files in os.walk(DATASET_DIR):
        # 该函数返回文件夹路径(root)、文件夹名(dirs)、文件名(files)
        for sub_dir in dirs:

            imgs = os.listdir(os.path.join(root, sub_dir))
            # 过滤出图像
            imgs = list(filter(lambda x: x.endswith(".png"), imgs))
            random.shuffle(imgs)
            img_count = len(imgs)

            train_point = int(img_count * TRAIN_PCT)
            valid_point = int(img_count * (TRAIN_PCT + VALID_PCT))

            for i in range(img_count):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_dir)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = os.path.join(test_dir, sub_dir)

                makedir(out_dir)

                target_path = os.path.join(out_dir, imgs[i])
                src_path = os.path.join(DATASET_DIR, sub_dir, imgs[i])

                shutil.copy(src_path, target_path)

            print(
                "Class:{}, train:{}, valid:{}, test:{}".format(
                    sub_dir,
                    train_point,
                    valid_point - train_point,
                    img_count - valid_point,
                )
            )
