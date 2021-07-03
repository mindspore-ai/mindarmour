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
"""
Membership Inference
"""

from multiprocessing import cpu_count
import numpy as np

import mindspore as ms
from mindspore.train import Model
from mindspore.dataset.engine import Dataset
from mindspore import Tensor
from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_param_type, check_param_multi_types, \
    check_model, check_numpy_param
from .attacker import get_attack_model
from ._check_config import verify_config_params

LOGGER = LogUtil.get_instance()
TAG = "MembershipInference"


def _eval_info(pred, truth, option):
    """
    Calculate the performance according to pred and truth.

    Args:
        pred (numpy.ndarray): Predictions for each sample.
        truth (numpy.ndarray): Ground truth for each sample.
        option (str): Type of evaluation indicators; Possible
            values are 'precision', 'accuracy' and 'recall'.

    Returns:
        float32, calculated evaluation results.

    Raises:
        ValueError, size of parameter pred or truth is 0.
        ValueError, value of parameter option must be in ["precision", "accuracy", "recall"].
    """
    check_numpy_param("pred", pred)
    check_numpy_param("truth", truth)

    if option == "accuracy":
        count = np.sum(pred == truth)
        return count / len(pred)
    if option == "precision":
        if np.sum(pred) == 0:
            return -1
        count = np.sum(pred & truth)
        return count / np.sum(pred)
    if option == "recall":
        if np.sum(truth) == 0:
            return -1
        count = np.sum(pred & truth)
        return count / np.sum(truth)

    msg = "The metric value {} is undefined.".format(option)
    LOGGER.error(TAG, msg)
    raise ValueError(msg)


def _softmax_cross_entropy(logits, labels, epsilon=1e-12):
    """
    Calculate the SoftmaxCrossEntropy result between logits and labels.

    Args:
        logits (numpy.ndarray): Numpy array of shape(N, C).
        labels (numpy.ndarray): Numpy array of shape(N, ).
        epsilon (float): The calculated value of softmax will be clipped into [epsilon, 1 - epsilon]. Default: 1e-12.

    Returns:
        numpy.ndarray: numpy array of shape(N, ), containing loss value for each vector in logits.
    """
    labels = np.eye(logits.shape[1])[labels].astype(np.int32)

    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    predictions = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    loss = -1 * np.sum(labels*np.log(predictions), axis=-1)
    return loss


class MembershipInference:
    """
    Evaluation proposed by Shokri, Stronati, Song and Shmatikov is a grey-box attack.
    The attack requires loss or logits results of training samples.

    References: `Reza Shokri, Marco Stronati, Congzheng Song, Vitaly Shmatikov.
    Membership Inference Attacks against Machine Learning Models. 2017.
    <https://arxiv.org/abs/1610.05820v2>`_

    Args:
        model (Model): Target model.
        n_jobs (int): Number of jobs run in parallel. -1 means using all processors,
            otherwise the value of n_jobs must be a positive integer.

    Examples:
        >>> # train_1, train_2 are non-overlapping datasets from training dataset of target model.
        >>> # test_1, test_2 are non-overlapping datasets from test dataset of target model.
        >>> # We use train_1, test_1 to train attack model, and use train_2, test_2 to evaluate attack model.
        >>> model = Model(network=net, loss_fn=loss, optimizer=opt, metrics={'acc', 'loss'})
        >>> attack_model = MembershipInference(model, n_jobs=-1)
        >>> config = [{"method": "KNN", "params": {"n_neighbors": [3, 5, 7]}}]
        >>> attack_model.train(train_1, test_1, config)
        >>> metrics = ["precision", "recall", "accuracy"]
        >>> result = attack_model.eval(train_2, test_2, metrics)

    Raises:
        TypeError: If type of model is not mindspore.train.Model.
        TypeError: If type of n_jobs is not int.
        ValueError: The value of n_jobs is neither -1 nor a positive integer.
    """

    def __init__(self, model, n_jobs=-1):
        check_param_type("n_jobs", n_jobs, int)
        if not (n_jobs == -1 or n_jobs > 0):
            msg = "Value of n_jobs must be either -1 or positive integer, but got {}.".format(n_jobs)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

        self._model = check_model("model", model, Model)
        self._n_jobs = min(n_jobs, cpu_count())
        self._attack_list = []

    def train(self, dataset_train, dataset_test, attack_config):
        """
        Depending on the configuration, use the input dataset to train the attack model.
        Save the attack model to self._attack_list.

        Args:
            dataset_train (mindspore.dataset): The training dataset for the target model.
            dataset_test (mindspore.dataset): The test set for the target model.
            attack_config (Union[list, tuple]): Parameter setting for the attack model. The format is
                [{"method": "knn", "params": {"n_neighbors": [3, 5, 7]}},
                {"method": "lr", "params": {"C": np.logspace(-4, 2, 10)}}].
                The support methods are knn, lr, mlp and rf, and the params of each method
                must within the range of changeable parameters. Tips of params implement
                can be found below:
                `KNN <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_,
                `LR <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_,
                `RF <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_,
                `MLP <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html>`_.

        Raises:
            KeyError: If any config in attack_config doesn't have keys {"method", "params"}.
            NameError: If the method(case insensitive) in attack_config is not in ["lr", "knn", "rf", "mlp"].
        """
        check_param_type("dataset_train", dataset_train, Dataset)
        check_param_type("dataset_test", dataset_test, Dataset)
        check_param_multi_types("attack_config", attack_config, (list, tuple))
        verify_config_params(attack_config)

        features, labels = self._transform(dataset_train, dataset_test)
        for config in attack_config:
            self._attack_list.append(get_attack_model(features, labels, config, n_jobs=self._n_jobs))


    def eval(self, dataset_train, dataset_test, metrics):
        """
        Evaluate the different privacy of the target model.
        Evaluation indicators shall be specified by metrics.

        Args:
            dataset_train (mindspore.dataset): The training dataset for the target model.
            dataset_test (mindspore.dataset): The test dataset for the target model.
            metrics (Union[list, tuple]): Evaluation indicators. The value of metrics
                must be in ["precision", "accuracy", "recall"]. Default: ["precision"].

        Returns:
            list, each element contains an evaluation indicator for the attack model.
        """
        check_param_type("dataset_train", dataset_train, Dataset)
        check_param_type("dataset_test", dataset_test, Dataset)
        check_param_multi_types("metrics", metrics, (list, tuple))

        metrics = set(metrics)
        metrics_list = {"precision", "accuracy", "recall"}
        if not metrics <= metrics_list:
            msg = "Element in 'metrics' must be in {}, but got {}.".format(metrics_list, metrics)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

        result = []
        features, labels = self._transform(dataset_train, dataset_test)
        for attacker in self._attack_list:
            pred = attacker.predict(features)
            item = {}
            for option in metrics:
                item[option] = _eval_info(pred, labels, option)
            result.append(item)
        return result

    def _transform(self, dataset_train, dataset_test):
        """
        Generate corresponding loss_logits features and new label, and return after shuffle.

        Args:
            dataset_train (mindspore.dataset): The train set for the target model.
            dataset_test (mindspore.dataset): The test set for the target model.

        Returns:
            - numpy.ndarray, loss_logits features for each sample. Shape is (N, C).
                N is the number of sample. C = 1 + dim(logits).
            - numpy.ndarray, labels for each sample, Shape is (N,).
        """
        features_train, labels_train = self._generate(dataset_train, 1)
        features_test, labels_test = self._generate(dataset_test, 0)
        features = np.vstack((features_train, features_test))
        labels = np.hstack((labels_train, labels_test))
        shuffle_index = np.array(range(len(labels)))
        np.random.shuffle(shuffle_index)
        features = features[shuffle_index]
        labels = labels[shuffle_index]

        return features, labels

    def _generate(self, input_dataset, label):
        """
        Return a loss_logits features and labels for training attack model.

        Args:
            input_dataset (mindspore.dataset): The dataset to be generated.
            label (int): Whether input_dataset belongs to the target model.

        Returns:
            - numpy.ndarray, loss_logits features for each sample. Shape is (N, C).
                N is the number of sample. C = 1 + dim(logits).

            - numpy.ndarray, labels for each sample, Shape is (N,).
        """
        loss_logits = np.array([])
        for batch in input_dataset.create_tuple_iterator(output_numpy=True):
            batch_data = Tensor(batch[0], ms.float32)
            batch_labels = batch[1].astype(np.int32)
            batch_logits = self._model.predict(batch_data).asnumpy()
            batch_loss = _softmax_cross_entropy(batch_logits, batch_labels)

            batch_feature = np.hstack((batch_loss.reshape(-1, 1), batch_logits))
            if loss_logits.size == 0:
                loss_logits = batch_feature
            else:
                loss_logits = np.vstack((loss_logits, batch_feature))

        if label == 1:
            labels = np.ones(len(loss_logits), np.int32)
        elif label == 0:
            labels = np.zeros(len(loss_logits), np.int32)
        else:
            msg = "The value of label must be 0 or 1, but got {}.".format(label)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        return loss_logits, labels
