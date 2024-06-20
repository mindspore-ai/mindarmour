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
"""implementation of CryptoClient and CryptoServer"""
import copy

import mindspore.nn as mspnn
import numpy as np
from nn.smc import Secure2PCClient
from nn.smc import Secure2PCServer
from nn.utils import initiate_client_model


class CryptoClient(mspnn.Cell):
    """Client for cryptographic operations in a secure federated learning system."""
    def __init__(self, model='mlp64', n_output=10, smc=None):
        super(CryptoClient, self).__init__()
        self.model = model
        self.n_output = n_output
        self.smc = smc
        self.encoder = initiate_client_model(self.model)

    @staticmethod
    def _add_bias_unit(x, how='column'):
        """
        Add bias unit (column or row of 1s) to the data array.

        Args:
            x (ndarray): Data array.
            how (str): 'column' or 'row', determines where to add the bias unit.

        Returns:
            ndarray: Data array with bias unit added.
        """
        if how == 'column':
            x_new = np.ones((x.shape[0], x.shape[1] + 1))
            x_new[:, 1:] = x
        elif how == 'row':
            x_new = np.ones((x.shape[0] + 1, x.shape[1]))
            x_new[1:, :] = x
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return x_new

    @staticmethod
    def _encode_labels(y, k):
        """
        Encode labels into a one-hot representation.

        Args:
            y (ndarray): Labels array.
            k (int): Number of classes.

        Returns:
            ndarray: One-hot encoded labels.
        """
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def construct(self, x):
        return self.encoder(x)

    def encrypt(self, intermediate, y):
        """
        Encrypt the data using secure multiparty computation.

        Args:
            x (ndarray): Data array to encrypt.

        Returns:
            Encrypted data.
        """
        intermediate_data = copy.deepcopy(intermediate)
        y_data = copy.deepcopy(y)

        y_onehot = self._encode_labels(y_data, self.n_output)
        intermediate_data = self._add_bias_unit(intermediate_data, how='column')

        if self.smc and isinstance(self.smc, Secure2PCClient):
            ct_feedforward = np.array(self.smc.execute_ndarray(intermediate_data))
            ct_backpropagation = np.array(self.smc.execute_ndarray(intermediate_data.T))
            return ct_feedforward, ct_backpropagation, y_onehot
        return intermediate_data, y_onehot


class CryptoServer:
    """Server for cryptographic operations in a secure federated learning system."""

    def __init__(self, n_features, hidden_layers, n_output=10,
                 l1=0.0, l2=0.0, eta=0.001,
                 smc=None, precision=None):
        self.n_output = n_output
        self.n_features = n_features
        self.hidden_layers = hidden_layers
        self.w = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.eta = eta
        self.smc = smc
        self.precision = precision

    def _initialize_weights(self):
        self.layers = [self.n_features] + self.hidden_layers + [self.n_output]
        w = [self._xavier_uniform((self.layers[i + 1], self.layers[i] + 1))
             for i in range(len(self.layers) - 1)]
        return w

    @staticmethod
    def _xavier_uniform(shape):
        fan_in, fan_out = shape[0], shape[1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        weights = np.random.uniform(low=-limit, high=limit, size=shape)
        return weights

    @staticmethod
    def _softmax(x):
        max_vals = np.max(x, axis=0, keepdims=True)
        e_x = np.exp(x - max_vals)
        return e_x / e_x.sum(axis=0, keepdims=True)

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _relu_gradient(z):
        return z > 0

    @staticmethod
    def _add_bias_unit(x, how='column'):
        """
        Add bias unit (column or row of 1s) to the data array.

        Args:
            x (ndarray): Data array.
            how (str): 'column' or 'row', determines where to add the bias unit.

        Returns:
            ndarray: Data array with bias unit added.
        """
        if how == 'column':
            x_new = np.ones((x.shape[0], x.shape[1] + 1))
            x_new[:, 1:] = x
        elif how == 'row':
            x_new = np.ones((x.shape[0] + 1, x.shape[1]))
            x_new[1:, :] = x
        else:
            raise AttributeError('`how` must be `column` or `row`')
        return x_new

    def feedforward(self, x, w):
        """
        Perform feedforward computation.

        Args:
            x (ndarray): Input data array.

        Returns:
            tuple: Activations and linear transformations.
        """
        z = [None for _ in range(len(w))]
        a = [None for _ in range(len(w))]
        if self.precision:
            z[0] = (w[0] * pow(10, self.precision)).astype(int).dot(x.T) / pow(10, self.precision)
        else:
            z[0] = w[0].dot(x.T)
        a[0] = self._add_bias_unit(self._relu(z[0]), how='row')
        for i in range(1, len(w)):
            z[i] = w[i].dot(a[i - 1])
            if i != len(w) - 1:
                a[i] = self._add_bias_unit(self._relu(z[i]), how='row')
            else:
                a[i] = self._softmax(z[i])
        return z, a

    def feedforward_secure(self, ct_batch, w):
        """
        Perform secure feedforward computation.

        Args:
            ct_batch: Encrypted batch of data.
            w: Weights for computation.

        Returns:
            tuple: Secure activations and linear transformations.
        """
        z = [None for _ in range(len(w))]
        a = [None for _ in range(len(w))]

        if isinstance(self.smc, Secure2PCServer):
            sk_w0 = self.smc.request_key_ndarray(w[0])
            z[0] = self.smc.execute_ndarray(sk_w0, ct_batch.tolist(), w[0])

        a[0] = self._add_bias_unit(self._relu(z[0]), how='row')

        for i in range(1, len(w)):
            z[i] = w[i].dot(a[i - 1])
            if i != len(w) - 1:
                a[i] = self._add_bias_unit(self._relu(z[i]), how='row')
            else:
                a[i] = self._softmax(z[i])
        return z, a

    def get_gradient(self, x, y_encode, a, z, w):
        """
        Compute gradient for backpropagation.

        Args:
            y_encode: Encoded labels.
            a: Activations.
            z: Linear transformations.
            w: Weights.

        Returns:
            tuple: Gradients and delta inputs.
        """
        sigma = [None for i in range(len(w))]
        grad = [None for i in range(len(w))]
        sigma[-1] = a[-1] - y_encode
        for i in range(len(w) - 2, -1, -1):
            sigma[i] = w[i + 1].T.dot(sigma[i + 1]) * self._relu_gradient(self._add_bias_unit(z[i], how='row'))
            sigma[i] = sigma[i][1:, :]
        if self.precision:
            grad[0] = (sigma[0] * pow(10, self.precision)).astype(int).dot(x) / pow(10, self.precision)
        else:
            grad[0] = sigma[0].dot(x)

        for i in range(1, len(w)):
            grad[i] = sigma[i].dot(a[i - 1].T)

        for i in range(len(w)):
            grad[i][:, 1:] += self.l2 * w[i][:, 1:]
            grad[i][:, 1:] += self.l1 * np.sign(w[i][:, 1:])

        d_inputs = w[0].T.dot(sigma[0])
        d_inputs = d_inputs[1:, :].T

        return grad, d_inputs

    def get_gradient_secure(self, ct_batch, y_encode, a, z, w):
        """
        Compute secure gradient for backpropagation.

        Args:
            ct_batch: Encrypted batch of data.
            y_encode: Encoded labels.
            a: Activations.
            z: Linear transformations.
            w: Weights.

        Returns:
            tuple: Secure gradients and delta inputs.
        """
        sigma = [None for _ in range(len(w))]
        grad = [None for _ in range(len(w))]
        sigma[-1] = a[-1] - y_encode
        for i in range(len(w) - 2, -1, -1):
            sigma[i] = w[i + 1].T.dot(sigma[i + 1]) * self._relu_gradient(self._add_bias_unit(z[i], how='row'))
            sigma[i] = sigma[i][1:, :]

        if isinstance(self.smc, Secure2PCServer):
            sk_sigma0 = self.smc.request_key_ndarray(sigma[0])
            grad[0] = self.smc.execute_ndarray(sk_sigma0, ct_batch.tolist(), sigma[0])

        for i in range(1, len(w)):
            grad[i] = sigma[i].dot(a[i - 1].T)

        for i in range(len(w)):
            grad[i][:, 1:] += self.l2 * w[i][:, 1:]
            grad[i][:, 1:] += self.l1 * np.sign(w[i][:, 1:])

        d_inputs = w[0].T.dot(sigma[0])
        d_inputs = d_inputs[1:, :].T

        return grad, d_inputs

    def predict(self, x):
        """
        Predict class labels for samples in x.

        Args:
            x (ndarray): Data array.

        Returns:
            ndarray: Predicted class labels.
        """
        if len(x.shape) != 2:
            raise AttributeError('X must be a [n_samples, n_features] array.\n'
                                 'Use X[:,None] for 1-feature classification,'
                                 '\nor X[[i]] for 1-sample classification')
        x = self._add_bias_unit(x, how='column')
        z, _ = self.feedforward(x, self.w)
        y_pred = np.argmax(z[len(z) - 1], axis=0)
        return y_pred
