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
Pointwise-Attack.
"""
import numpy as np

from mindarmour.utils._check_param import check_model, check_pair_numpy_param, \
    check_int_positive, check_param_type
from mindarmour.utils.logger import LogUtil
from ..attack import Attack
from .black_model import BlackModel
from .salt_and_pepper_attack import SaltAndPepperNoiseAttack

LOGGER = LogUtil.get_instance()
TAG = 'PointWiseAttack'


class PointWiseAttack(Attack):
    """
    The Pointwise Attack make sure use the minimum number of changed pixels to generate adversarial sample for each
    original sample.Those changed pixels will use binary search to make sure the distance between adversarial sample
    and original sample is as close as possible.

    References: `L. Schott, J. Rauber, M. Bethge, W. Brendel: "Towards the
    first adversarially robust neural network model on MNIST", ICLR (2019)
    <https://arxiv.org/abs/1805.09190>`_

    Args:
        model (BlackModel): Target model.
        max_iter (int): Max rounds of iteration to generate adversarial image. Default: 1000.
        search_iter (int): Max rounds of binary search. Default: 10.
        is_targeted (bool): If True, targeted attack. If False, untargeted attack. Default: False.
        init_attack (Attack): Attack used to find a starting point. Default: None.
        sparse (bool): If True, input labels are sparse-encoded. If False, input labels are one-hot-encoded.
            Default: True.

    Examples:
        >>> attack = PointWiseAttack(model)
    """

    def __init__(self, model, max_iter=1000, search_iter=10, is_targeted=False, init_attack=None, sparse=True):
        super(PointWiseAttack, self).__init__()
        self._model = check_model('model', model, BlackModel)
        self._max_iter = check_int_positive('max_iter', max_iter)
        self._search_iter = check_int_positive('search_iter', search_iter)
        self._is_targeted = check_param_type('is_targeted', is_targeted, bool)
        if init_attack is None:
            self._init_attack = SaltAndPepperNoiseAttack(model, is_targeted=self._is_targeted)
        else:
            self._init_attack = init_attack
        self._sparse = check_param_type('sparse', sparse, bool)

    def generate(self, inputs, labels):
        """
        Generate adversarial examples based on input samples and targeted labels.

        Args:
            inputs (numpy.ndarray): Benign input samples used as references to create adversarial examples.
            labels (numpy.ndarray): For targeted attack, labels are adversarial target labels.
                For untargeted attack, labels are ground-truth labels.

        Returns:
            - numpy.ndarray, bool values for each attack result.

            - numpy.ndarray, generated adversarial examples.

            - numpy.ndarray, query times for each sample.

        Examples:
            >>> is_adv_list, adv_list, query_times_each_adv = attack.generate([[0.1, 0.2, 0.6], [0.3, 0, 0.4]], [2, 3])
        """
        arr_x, arr_y = check_pair_numpy_param('inputs', inputs, 'labels', labels)
        if not self._sparse:
            arr_y = np.argmax(arr_y, axis=1)
        ini_bool, ini_advs, ini_count = self._initialize_starting_point(arr_x, arr_y)
        is_adv_list = list()
        adv_list = list()
        query_times_each_adv = list()
        for sample, sample_label, start_adv, ite_bool, ite_c in zip(arr_x, arr_y, ini_advs, ini_bool, ini_count):
            if ite_bool:
                LOGGER.info(TAG, 'Start optimizing.')
                ori_label = np.argmax(self._model.predict(np.expand_dims(sample, axis=0))[0])
                ini_label = np.argmax(self._model.predict(np.expand_dims(start_adv, axis=0))[0])
                is_adv, adv_x, query_times = self._decision_optimize(sample, sample_label, start_adv)
                adv_label = np.argmax(self._model.predict(np.expand_dims(adv_x, axis=0))[0])
                LOGGER.info(TAG, 'before ini attack label is :{}'.format(ori_label))
                LOGGER.info(TAG, 'after ini attack label is :{}'.format(ini_label))
                LOGGER.info(TAG, 'INPUT optimize label is :{}'.format(sample_label))
                LOGGER.info(TAG, 'after pointwise attack label is :{}'.format(adv_label))
                is_adv_list.append(is_adv)
                adv_list.append(adv_x)
                query_times_each_adv.append(query_times + ite_c)
            else:
                LOGGER.info(TAG, 'Initial sample is not adversarial, pass.')
                is_adv_list.append(False)
                adv_list.append(start_adv)
                query_times_each_adv.append(ite_c)
        is_adv_list = np.array(is_adv_list)
        adv_list = np.array(adv_list)
        query_times_each_adv = np.array(query_times_each_adv)
        LOGGER.info(TAG, 'ret list is: {}'.format(adv_list))
        return is_adv_list, adv_list, query_times_each_adv

    def _decision_optimize(self, unperturbed_img, input_label, perturbed_img):
        """
        Make the perturbed samples more similar to unperturbed samples,
        while maintaining the perturbed_label.

        Args:
            unperturbed_img (numpy.ndarray): Input sample as reference to create
                adversarial example.
            input_label (numpy.ndarray): Input label.
            perturbed_img (numpy.ndarray): Starting point to optimize.

        Returns:
            numpy.ndarray, a generated adversarial example.

        Raises:
            ValueError: if input unperturbed and perturbed samples have different size.
        """
        query_count = 0
        img_size = unperturbed_img.size
        img_shape = unperturbed_img.shape
        perturbed_img = perturbed_img.reshape(-1)
        unperturbed_img = unperturbed_img.reshape(-1)
        recover = np.copy(perturbed_img)

        if unperturbed_img.dtype != perturbed_img.dtype:
            msg = 'unperturbed sample and perturbed sample must have the same' \
                  ' dtype, but got dtype of unperturbed is: {}, dtype of perturbed ' \
                  'is: {}'.format(unperturbed_img.dtype, perturbed_img.dtype)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        l2_dis = np.linalg.norm(perturbed_img - unperturbed_img)
        LOGGER.info(TAG, 'Before optimize, the l2 distance between original ' \
                         'sample and adversarial sample is: {}'.format(l2_dis))
        # recover pixel if image is adversarial
        for _ in range(self._max_iter):
            is_improve = False
            # at the premise of adversarial feature, recover pixels
            pixels_ind = np.arange(img_size)
            mask = unperturbed_img != perturbed_img
            np.random.shuffle(pixels_ind)
            for ite_ind in pixels_ind:
                if mask[ite_ind]:
                    recover[ite_ind] = unperturbed_img[ite_ind]
                    query_count += 1
                    is_adv = self._model.is_adversarial(recover.reshape(img_shape), input_label, self._is_targeted)
                    if is_adv:
                        is_improve = True
                        perturbed_img[ite_ind] = recover[ite_ind]
                        break
                    else:
                        recover[ite_ind] = perturbed_img[ite_ind]
            l2_dis = np.linalg.norm(perturbed_img - unperturbed_img)
            if not is_improve or (np.square(l2_dis) / np.sqrt(len(pixels_ind)) <= self._get_threthod()):
                break
        LOGGER.debug(TAG, 'first round: Query count {}'.format(query_count))
        LOGGER.debug(TAG, 'Starting binary searches.')
        # tag the optimized pixels.
        mask = unperturbed_img != perturbed_img
        for _ in range(self._max_iter):
            is_improve = False
            pixels_ind = np.arange(img_size)
            np.random.shuffle(pixels_ind)
            for ite_ind in pixels_ind:
                if not mask[ite_ind]:
                    continue
                recover[ite_ind] = unperturbed_img[ite_ind]
                query_count += 1
                is_adv = self._model.is_adversarial(recover.reshape(img_shape), input_label, self._is_targeted)
                if is_adv:
                    is_improve = True
                    mask[ite_ind] = False
                    perturbed_img[ite_ind] = recover[ite_ind]
                    l2_dis = np.linalg.norm(perturbed_img - unperturbed_img)
                    LOGGER.info(TAG, 'Reset {}th pixel value to original, l2 distance: {}.'.format(ite_ind, l2_dis))
                    break
                else:
                    # use binary searches
                    optimized_value, b_query = self._binary_search(perturbed_img,
                                                                   unperturbed_img,
                                                                   ite_ind,
                                                                   input_label, img_shape)
                    query_count += b_query
                    if optimized_value != perturbed_img[ite_ind]:
                        is_improve = True
                        perturbed_img[ite_ind] = optimized_value
                        l2_dis = np.linalg.norm(perturbed_img - unperturbed_img)
                        LOGGER.info(TAG, 'Reset {}th pixel value to original, l2 distance: {}.'.format(ite_ind, l2_dis))
                        break
            l2_dis = np.linalg.norm(perturbed_img - unperturbed_img)
            if not is_improve or (np.square(l2_dis) / np.sqrt(len(pixels_ind)) <= self._get_threthod()):
                LOGGER.info(TAG, 'second optimized finish.')
                break
        LOGGER.info(TAG, 'Optimized finished, query count is {}'.format(query_count))
        # this method use to optimized the adversarial sample
        return True, perturbed_img.reshape(img_shape), query_count

    def _binary_search(self, perturbed_img, unperturbed_img, ite_ind, input_label, img_shape):
        """
        For original pixel of inputs, use binary search to get the nearest pixel
        value with original value with adversarial feature.

        Args:
            perturbed_img (numpy.ndarray): Adversarial sample.
            unperturbed_img (numpy.ndarray): Input sample.
            ite_ind (int): The index of pixel in inputs.
            input_label (numpy.ndarray): Input labels.
            img_shape (tuple): Shape of the original sample.

        Returns:
            float, adversarial pixel value.
        """
        query_count = 0
        adv_value = perturbed_img[ite_ind]
        non_adv_value = unperturbed_img[ite_ind]
        for _ in range(self._search_iter):
            next_value = (adv_value + non_adv_value) / 2
            recover = np.copy(perturbed_img)
            recover[ite_ind] = next_value
            query_count += 1
            is_adversarial = self._model.is_adversarial(
                recover.reshape(img_shape), input_label, self._is_targeted)
            if is_adversarial:
                adv_value = next_value
            else:
                non_adv_value = next_value
        return adv_value, query_count

    def _initialize_starting_point(self, inputs, labels):
        """
        Use init_attack to generate original adversarial inputs.

        Args:
            inputs (numpy.ndarray): Benign input sample used as references to create
                adversarial examples.
            labels (numpy.ndarray): If is targeted attack, labels is adversarial
                labels, if is untargeted attack, labels is true labels.

        Returns:
            numpy.ndarray, adversarial image(s) generate by init_attack method.
        """
        is_adv, start_adv, query_c = self._init_attack.generate(inputs, labels)
        return is_adv, start_adv, query_c

    def _get_threthod(self):
        """
        Return a float number, when distance small than this number,
        optimize will abort early.

        Returns:
            float, the optimized level, the smaller of number, the better
            of adversarial sample.
        """
        predefined_threshold = 0.01
        return predefined_threshold
