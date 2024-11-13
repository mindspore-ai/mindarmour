# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""function of initiation in client model and server model"""
import mindspore.nn as mspnn


def initiate_client_model(model):
    """
    Initialize the client model based on the given model name.

    Args:
        model (str): The name of the model to initialize.

    Returns:
        mindspore.nn.Cell: The initialized client model.

    Raises:
        NameError: If the model name is not recognized.
    """
    if model == 'mlp64':
        net = mspnn.SequentialCell(
            mspnn.Flatten(),
            mspnn.Dense(784, 128),
            mspnn.ReLU(),
            mspnn.Dense(128, 64),
            mspnn.ReLU(),
        )
        return net
    raise NameError('choose model from mlp64')


def initiate_server_model(model):
    """
    Initialize the server model based on the given model name.

    Args:
        model (str): The name of the model to initialize.

    Returns:
        tuple: The number of features and a list of hidden layers.

    Raises:
        NameError: If the model name is not recognized.
    """
    if model == 'mlp64':
        n_features = 64
        hidden_layers = [32, 16]
        return n_features, hidden_layers
    raise NameError('choose model!')
