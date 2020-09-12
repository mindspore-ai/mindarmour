# Copyright 2020 Huawei Technologies Co., Ltd
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
Examples of membership inference
"""
import argparse
import sys
import numpy as np

from mindspore.train import Model
from mindspore.train.serialization import load_param_into_net, load_checkpoint
import mindspore.nn as nn
from mindarmour.privacy.evaluation import MembershipInference
from mindarmour.utils import LogUtil

from examples.common.networks.vgg.vgg import vgg16
from examples.common.networks.vgg.config import cifar_cfg as cfg
from examples.common.networks.vgg.utils.util import get_param_groups
from examples.common.dataset.data_processing import vgg_create_dataset100

logging = LogUtil.get_instance()
logging.set_level(20)

sys.path.append("../../../")

TAG = "membership inference example"


if __name__ == "__main__":
    parser = argparse.ArgumentParser("main case arg parser.")
    parser.add_argument("--device_target", type=str, default="Ascend",
                        choices=["Ascend"])
    parser.add_argument("--data_path", type=str, required=True,
                        help="Data home path for Cifar100.")
    parser.add_argument("--pre_trained", type=str, required=True,
                        help="Checkpoint path.")
    args = parser.parse_args()
    args.num_classes = cfg.num_classes
    args.batch_norm = cfg.batch_norm
    args.has_dropout = cfg.has_dropout
    args.has_bias = cfg.has_bias
    args.initialize_mode = cfg.initialize_mode
    args.padding = cfg.padding
    args.pad_mode = cfg.pad_mode
    args.weight_decay = cfg.weight_decay
    args.loss_scale = cfg.loss_scale

    # load the pretrained model
    net = vgg16(args.num_classes, args)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    opt = nn.Momentum(params=get_param_groups(net), learning_rate=0.1, momentum=0.9,
                      weight_decay=args.weight_decay, loss_scale=args.loss_scale)
    load_param_into_net(net, load_checkpoint(args.pre_trained))
    model = Model(network=net, loss_fn=loss, optimizer=opt)
    logging.info(TAG, "The model is loaded.")
    attacker = MembershipInference(model)
    config = [
        {
            "method": "knn",
            "params": {
                "n_neighbors": [3, 5, 7]
            }
        },
        {
            "method": "lr",
            "params": {
                "C": np.logspace(-4, 2, 10)
            }
        },
        {
            "method": "mlp",
            "params": {
                "hidden_layer_sizes": [(64,), (32, 32)],
                "solver": ["adam"],
                "alpha": [0.0001, 0.001, 0.01]
            }
        },
        {
            "method": "rf",
            "params": {
                "n_estimators": [100],
                "max_features": ["auto", "sqrt"],
                "max_depth": [5, 10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        }
    ]

    # load and split dataset
    train_dataset = vgg_create_dataset100(data_home=args.data_path, image_size=(224, 224),
                                          batch_size=64, num_samples=10000, shuffle=False)
    test_dataset = vgg_create_dataset100(data_home=args.data_path, image_size=(224, 224),
                                         batch_size=64, num_samples=10000, shuffle=False, training=False)
    train_train, eval_train = train_dataset.split([0.8, 0.2])
    train_test, eval_test = test_dataset.split([0.8, 0.2])
    logging.info(TAG, "Data loading is complete.")

    logging.info(TAG, "Start training the inference model.")
    attacker.train(train_train, train_test, config)
    logging.info(TAG, "The inference model is training complete.")

    logging.info(TAG, "Start the evaluation phase")
    metrics = ["precision", "accuracy", "recall"]
    result = attacker.eval(eval_train, eval_test, metrics)

    # Show the metrics for each attack method.
    count = len(config)
    for i in range(count):
        print("Method: {}, {}".format(config[i]["method"], result[i]))
