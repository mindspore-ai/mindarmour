# Copyright 2019 Huawei Technologies Co., Ltd
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
Natural-evolutionary-strategy Attack.
"""
import time
import numpy as np
from scipy.special import softmax

from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_pair_numpy_param, check_model, \
    check_numpy_param, check_int_positive, check_value_positive, check_param_type
from ..attack import Attack
from .black_model import BlackModel

LOGGER = LogUtil.get_instance()
TAG = 'NES'


def _bound(image, epislon):
    lower = np.clip(image - epislon, 0, 1)
    upper = np.clip(image + epislon, 0, 1)
    return lower, upper


class NES(Attack):
    """
    The class is an implementation of the Natural Evolutionary Strategies Attack
    Method. NES uses natural evolutionary strategies to estimate gradients to
    improve query efficiency. NES covers three settings: Query-Limited setting,
    Partial-Information setting and Label-Only setting. In the query-limit
    setting, the attack has a limited number of queries to the target model but
    access to the probabilities of all classes. In the partial-info setting,
    the attack only has access to the probabilities for top-k classes.
    In the label-only setting, the attack only has access to a list of k inferred
    labels ordered by their predicted probabilities. In the Partial-Information
    setting and Label-Only setting, NES do target attack so user need to use
    set_target_images method to set target images of target classes.

    References: `Andrew Ilyas, Logan Engstrom, Anish Athalye, and Jessy Lin.
    Black-box adversarial attacks with limited queries and information. In
    ICML, July 2018 <https://arxiv.org/abs/1804.08598>`_

    Args:
        model (BlackModel): Target model to be attacked.
        scene (str): Scene in 'Label_Only', 'Partial_Info' or 'Query_Limit'.
        max_queries (int): Maximum query numbers to generate an adversarial example. Default: 10000.
        top_k (int): For Partial-Info or Label-Only setting, indicating how much (Top-k) information is
            available for the attacker. For Query-Limited setting, this input should be set as -1. Default: -1.
        num_class (int): Number of classes in dataset. Default: 10.
        batch_size (int): Batch size. Default: 128.
        epsilon (float): Maximum perturbation allowed in attack. Default: 0.3.
        samples_per_draw (int): Number of samples draw in antithetic sampling. Default: 128.
        momentum (float): Momentum. Default: 0.9.
        learning_rate (float): Learning rate. Default: 1e-3.
        max_lr (float): Max Learning rate. Default: 5e-2.
        min_lr (float): Min Learning rate. Default: 5e-4.
        sigma (float): Step size of random noise. Default: 1e-3.
        plateau_length (int): Length of plateau used in Annealing algorithm. Default: 20.
        plateau_drop (float): Drop of plateau used in Annealing algorithm. Default: 2.0.
        adv_thresh (float): Threshold of adversarial. Default: 0.25.
        zero_iters (int): Number of points to use for the proxy score. Default: 10.
        starting_eps (float): Starting epsilon used in Label-Only setting. Default: 1.0.
        starting_delta_eps (float): Delta epsilon used in Label-Only setting. Default: 0.5.
        label_only_sigma (float): Sigma used in Label-Only setting. Default: 1e-3.
        conservative (int): Conservation used in epsilon decay, it will increase if no convergence. Default: 2.
        sparse (bool): If True, input labels are sparse-encoded. If False,
            input labels are one-hot-encoded. Default: True.

    Examples:
        >>> SCENE = 'Label_Only'
        >>> TOP_K = 5
        >>> num_class = 5
        >>> nes_instance = NES(user_model, SCENE, top_k=TOP_K)
        >>> initial_img = np.asarray(np.random.random((32, 32)), np.float32)
        >>> target_image  = np.asarray(np.random.random((32, 32)), np.float32)
        >>> orig_class = 0
        >>> target_class = 2
        >>> nes_instance.set_target_images(target_image)
        >>> tag, adv, queries = nes_instance.generate([initial_img], [target_class])
    """

    def __init__(self, model, scene, max_queries=10000, top_k=-1, num_class=10, batch_size=128, epsilon=0.3,
                 samples_per_draw=128, momentum=0.9, learning_rate=1e-3, max_lr=5e-2, min_lr=5e-4, sigma=1e-3,
                 plateau_length=20, plateau_drop=2.0, adv_thresh=0.25, zero_iters=10, starting_eps=1.0,
                 starting_delta_eps=0.5, label_only_sigma=1e-3, conservative=2, sparse=True):
        super(NES, self).__init__()
        self._model = check_model('model', model, BlackModel)
        self._scene = scene

        self._max_queries = check_int_positive('max_queries', max_queries)
        self._num_class = check_int_positive('num_class', num_class)
        self._batch_size = check_int_positive('batch_size', batch_size)
        self._samples_per_draw = check_int_positive('samples_per_draw', samples_per_draw)
        self._goal_epsilon = check_value_positive('epsilon', epsilon)
        self._momentum = check_value_positive('momentum', momentum)
        self._learning_rate = check_value_positive('learning_rate', learning_rate)
        self._max_lr = check_value_positive('max_lr', max_lr)
        self._min_lr = check_value_positive('min_lr', min_lr)
        self._sigma = check_value_positive('sigma', sigma)
        self._plateau_length = check_int_positive('plateau_length', plateau_length)
        self._plateau_drop = check_value_positive('plateau_drop', plateau_drop)
        # partial information arguments
        self._k = top_k
        self._adv_thresh = check_value_positive('adv_thresh', adv_thresh)
        # label only arguments
        self._zero_iters = check_int_positive('zero_iters', zero_iters)
        self._starting_eps = check_value_positive('starting_eps', starting_eps)
        self._starting_delta_eps = check_value_positive('starting_delta_eps', starting_delta_eps)
        self._label_only_sigma = check_value_positive('label_only_sigma', label_only_sigma)
        self._conservative = check_int_positive('conservative', conservative)
        self._sparse = check_param_type('sparse', sparse, bool)
        self.target_imgs = None
        self.target_img = None
        self.target_class = None

    def generate(self, inputs, labels):
        """
        Main algorithm for NES.

        Args:
            inputs (numpy.ndarray): Benign input samples.
            labels (numpy.ndarray): Target labels.

        Returns:
            - numpy.ndarray, bool values for each attack result.

            - numpy.ndarray, generated adversarial examples.

            - numpy.ndarray, query times for each sample.

        Raises:
            ValueError: If the top_k less than 0 in Label-Only or Partial-Info setting.
            ValueError: If the target_imgs is None in Label-Only or Partial-Info setting.
            ValueError: If scene is not in ['Label_Only', 'Partial_Info', 'Query_Limit']

        Examples:
            >>> advs = attack.generate([[0.2, 0.3, 0.4], [0.3, 0.3, 0.2]],
            >>> [1, 2])
        """
        inputs, labels = check_pair_numpy_param('inputs', inputs, 'labels', labels)
        if not self._sparse:
            labels = np.argmax(labels, axis=1)

        if self._scene == 'Label_Only' or self._scene == 'Partial_Info':
            if self._k < 1:
                msg = "In 'Label_Only' or 'Partial_Info' mode, 'top_k' must more than 0."
                LOGGER.error(TAG, msg)
                raise ValueError(msg)
            if self.target_imgs is None:
                msg = "In 'Label_Only' or 'Partial_Info' mode, 'target_imgs' must be set."
                LOGGER.error(TAG, msg)
                raise ValueError(msg)

        elif self._scene == 'Query_Limit':
            self._k = self._num_class
        else:
            msg = "scene must be string in 'Label_Only', 'Partial_Info' or 'Query_Limit' "
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

        is_advs = []
        advs = []
        queries = []
        for sample, label, target_img in zip(inputs, labels, self.target_imgs):
            is_adv, adv, query = self._generate_one(sample, label, target_img)
            is_advs.append(is_adv)
            advs.append(adv)
            queries.append(query)

        return is_advs, advs, queries

    def set_target_images(self, target_images):
        """
        Set target samples for target attack in the Partial-Info setting or Label-Only setting.

        Args:
            target_images (numpy.ndarray): Target samples for target attack.
        """
        self.target_imgs = check_numpy_param('target_images', target_images)

    def _generate_one(self, origin_image, target_label, target_image):
        """
        Main algorithm for NES.

        Args:
            origin_image (numpy.ndarray): Benign input sample.
            target_label (int): Target label.

        Returns:
            - bool.
                - If True: successfully make an adversarial example.

                - If False: unsuccessfully make an adversarial example.

            - numpy.ndarray, an adversarial example.

            - int, number of queries.
        """
        self.target_class = target_label
        origin_image = check_numpy_param('origin_image', origin_image)
        self._epsilon = self._starting_eps
        lower, upper = _bound(origin_image, self._epsilon)
        goal_epsilon = self._goal_epsilon
        delta_epsilon = self._starting_delta_eps
        if self._scene == 'Label_Only' or self._scene == 'Partial_Info':
            adv = target_image
        else:
            adv = origin_image.copy()

        # for backtracking and momentum
        num_queries = 0
        gradient = 0
        last_ls = []
        max_iters = int(np.ceil(self._max_queries // self._samples_per_draw))
        for i in range(max_iters):
            start = time.time()
            # early stop
            eval_preds = self._model.predict(adv)
            eval_preds = np.argmax(eval_preds, axis=1)
            padv = np.equal(eval_preds, self.target_class)
            if padv and self._epsilon <= goal_epsilon:
                LOGGER.debug(TAG, 'early stopping at iteration %d', i)
                return True, adv, num_queries

            #  antithetic sampling noise
            size = (self._batch_size // 2,) + origin_image.shape
            noise_pos = np.random.normal(size=size)
            noise = np.concatenate((noise_pos, -noise_pos), axis=0)
            eval_points = adv + self._sigma*noise

            prev_g = gradient
            loss, gradient = self._get_grad(origin_image, eval_points, noise)
            gradient = self._momentum*prev_g + (1.0 - self._momentum)*gradient

            # plateau learning rate annealing
            last_ls.append(loss)
            last_ls = self._plateau_annealing(last_ls)

            # search for learning rate and epsilon decay
            current_lr = self._max_lr
            prop_delta_eps = 0.0
            if loss < self._adv_thresh and self._epsilon > goal_epsilon:
                prop_delta_eps = delta_epsilon
            while current_lr >= self._min_lr:
                # in partial information only or label only setting
                if self._scene == 'Label_Only' or self._scene == 'Partial_Info':
                    proposed_epsilon = max(self._epsilon - prop_delta_eps, goal_epsilon)
                    lower, upper = _bound(origin_image, proposed_epsilon)
                proposed_adv = adv - current_lr*np.sign(gradient)
                proposed_adv = np.clip(proposed_adv, lower, upper)
                num_queries += 1

                if self._preds_in_top_k(self.target_class, proposed_adv):
                    # The predicted label of proposed adversarial examples is in
                    # the top k observations.
                    if prop_delta_eps > 0:
                        delta_epsilon = max(prop_delta_eps, 0.1)
                        last_ls = []
                    adv = proposed_adv
                    self._epsilon = self._epsilon - prop_delta_eps / self._conservative
                    self._epsilon = max(self._epsilon, goal_epsilon)
                    break
                elif current_lr >= self._min_lr*2:
                    current_lr = current_lr / 2
                    LOGGER.debug(TAG, "backtracking learning rate to %.3f", current_lr)
                else:
                    prop_delta_eps = prop_delta_eps / 2
                    if prop_delta_eps < 2e-3:
                        LOGGER.debug(TAG, "Did not converge.")
                        return False, adv, num_queries
                    current_lr = self._max_lr
                    LOGGER.debug(TAG, "backtracking epsilon to %.3f", self._epsilon - prop_delta_eps)

            # update the number of queries
            if self._scene == 'Label_Only':
                num_queries += self._samples_per_draw*self._zero_iters
            else:
                num_queries += self._samples_per_draw
            LOGGER.debug(TAG,
                         'Step %d: loss %.4f, lr %.2E, eps %.3f, time %.4f.',
                         i,
                         loss,
                         current_lr,
                         self._epsilon,
                         time.time() - start)

        return False, adv, num_queries

    def _plateau_annealing(self, last_loss):
        last_loss = last_loss[-self._plateau_length:]
        if last_loss[-1] > last_loss[0] and len(last_loss) == self._plateau_length:
            if self._max_lr > self._min_lr:
                LOGGER.debug(TAG, "Annealing max learning rate.")
                self._max_lr = max(self._max_lr / self._plateau_drop, self._min_lr)
            last_loss = []
        return last_loss

    def _softmax_cross_entropy_with_logit(self, logit):
        logit = softmax(logit, axis=1)
        onehot_label = np.zeros(self._num_class)
        onehot_label[self.target_class] = 1
        onehot_labels = np.tile(onehot_label, (len(logit), 1))
        entropy = -onehot_labels*np.log(logit)
        loss = np.mean(entropy, axis=1)
        return loss

    def _query_limit_loss(self, eval_points, noise):
        """
        Loss in Query-Limit setting.
        """
        LOGGER.debug(TAG, 'enter the function _query_limit_loss().')
        loss = self._softmax_cross_entropy_with_logit(self._model.predict(eval_points))

        return loss, noise

    def _partial_info_loss(self, eval_points, noise):
        """
        Loss in Partial-Info setting.
        """
        LOGGER.debug(TAG, 'enter the function _partial_info_loss.')
        logit = self._model.predict(eval_points)
        loss = np.sort(softmax(logit, axis=1))[:, -self._k:]
        inds = np.argsort(logit)[:, -self._k:]
        good_loss = np.where(np.equal(inds, self.target_class), loss, np.zeros(np.shape(inds)))
        good_loss = np.max(good_loss, axis=1)
        losses = -np.log(good_loss)
        return losses, noise

    def _label_only_loss(self, origin_image, eval_points, noise):
        """
        Loss in Label-Only setting.
        """
        LOGGER.debug(TAG, 'enter the function _label_only_loss().')
        tiled_points = np.tile(np.expand_dims(eval_points, 0), [self._zero_iters, *[1]*len(eval_points.shape)])
        noised_eval_im = tiled_points + np.random.randn(self._zero_iters,
                                                        self._batch_size,
                                                        *origin_image.shape)*self._label_only_sigma
        noised_eval_im = np.reshape(noised_eval_im, (self._zero_iters*self._batch_size, *origin_image.shape))
        logits = self._model.predict(noised_eval_im)
        inds = np.argsort(logits)[:, -self._k:]
        real_inds = np.reshape(inds, (self._zero_iters, self._batch_size, -1))
        rank_range = np.arange(1, self._k + 1, 1, dtype=np.float32)
        tiled_rank_range = np.tile(np.reshape(rank_range, (1, 1, self._k)), [self._zero_iters, self._batch_size, 1])
        batches_in = np.where(np.equal(real_inds, self.target_class),
                              tiled_rank_range,
                              np.zeros(np.shape(tiled_rank_range)))
        loss = 1 - np.mean(batches_in)
        return loss, noise

    def _preds_in_top_k(self, target_class, prop_adv_):
        # query limit setting
        if self._k == self._num_class:
            return True
        # label only and partial information setting
        eval_preds = self._model.predict(prop_adv_)
        if not target_class in eval_preds.argsort()[:, -self._k:]:
            return False
        return True

    def _get_grad(self, origin_image, eval_points, noise):
        """Calculate gradient."""
        losses = []
        grads = []
        for _ in range(self._samples_per_draw // self._batch_size):
            if self._scene == 'Label_Only':
                loss, np_noise = self._label_only_loss(origin_image, eval_points, noise)
            elif self._scene == 'Partial_Info':
                loss, np_noise = self._partial_info_loss(eval_points, noise)
            else:
                loss, np_noise = self._query_limit_loss(eval_points, noise)
            # only support three channel images
            losses_tiled = np.tile(np.reshape(loss, (-1, 1, 1, 1)), (1,) + origin_image.shape)
            grad = np.mean(losses_tiled*np_noise, axis=0) / self._sigma

            grads.append(grad)
            losses.append(np.mean(loss))
        return np.array(losses).mean(), np.mean(np.array(grads), axis=0)
