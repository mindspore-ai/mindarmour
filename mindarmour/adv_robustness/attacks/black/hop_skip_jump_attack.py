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
Hop-skip-jump attack.
"""
import numpy as np


from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_pair_numpy_param, check_model, \
    check_numpy_param, check_int_positive, check_value_positive, \
    check_value_non_negative, check_param_type
from ..attack import Attack
from .black_model import BlackModel

LOGGER = LogUtil.get_instance()
TAG = 'HopSkipJumpAttack'


def _clip_image(image, clip_min, clip_max):
    """
    Clip an image, or an image batch, with upper and lower threshold.
    """
    return np.clip(image, clip_min, clip_max)


class HopSkipJumpAttack(Attack):
    """
    HopSkipJumpAttack proposed by Chen, Jordan and Wainwright is a
    decision-based attack. The attack requires access to output labels of
    target model.

    References: `Chen J, Michael I. Jordan, Martin J. Wainwright.
    HopSkipJumpAttack: A Query-Efficient Decision-Based Attack. 2019.
    arXiv:1904.02144 <https://arxiv.org/abs/1904.02144>`_

    Args:
        model (BlackModel): Target model.
        init_num_evals (int): The initial number of evaluations for gradient
            estimation. Default: 100.
        max_num_evals (int): The maximum number of evaluations for gradient
            estimation. Default: 1000.
        stepsize_search (str): Indicating how to search for stepsize; Possible
            values are 'geometric_progression', 'grid_search', 'geometric_progression'.
            Default: 'geometric_progression'.
        num_iterations (int): The number of iterations. Default: 20.
        gamma (float): Used to set binary search threshold theta. Default: 1.0.
            For l2 attack the binary search threshold `theta` is
            :math:`gamma / d^{3/2}`. For linf attack is :math:`gamma / d^2`.
            Default: 1.0.
        constraint (str): The norm distance to optimize. Possible values are 'l2',
            'linf'. Default: l2.
        batch_size (int): Batch size. Default: 32.
        clip_min (float, optional): The minimum image component value.
            Default: 0.
        clip_max (float, optional): The maximum image component value.
            Default: 1.
        sparse (bool): If True, input labels are sparse-encoded. If False,
            input labels are one-hot-encoded. Default: True.

    Raises:
        ValueError: If stepsize_search not in ['geometric_progression',
            'grid_search']
        ValueError: If constraint not in ['l2', 'linf']

    Examples:
        >>> x_test = np.asarray(np.random.random((sample_num,
        >>> sample_length)), np.float32)
        >>> y_test = np.random.randint(0, class_num, size=sample_num)
        >>> instance = HopSkipJumpAttack(user_model)
        >>> adv_x = instance.generate(x_test, y_test)
    """

    def __init__(self, model, init_num_evals=100, max_num_evals=1000,
                 stepsize_search='geometric_progression', num_iterations=20,
                 gamma=1.0, constraint='l2', batch_size=32, clip_min=0.0,
                 clip_max=1.0, sparse=True):
        super(HopSkipJumpAttack, self).__init__()
        self._model = check_model('model', model, BlackModel)
        self._init_num_evals = check_int_positive('initial_num_evals',
                                                  init_num_evals)
        self._max_num_evals = check_int_positive('max_num_evals', max_num_evals)
        self._batch_size = check_int_positive('batch_size', batch_size)
        self._clip_min = check_value_non_negative('clip_min', clip_min)
        self._clip_max = check_value_non_negative('clip_max', clip_max)
        self._sparse = check_param_type('sparse', sparse, bool)
        self._np_dtype = np.dtype('float32')
        if stepsize_search in ['geometric_progression', 'grid_search']:
            self._stepsize_search = stepsize_search
        else:
            msg = "stepsize_search must be in ['geometric_progression'," \
                  " 'grid_search'], but got {}".format(stepsize_search)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

        self._num_iterations = check_int_positive('num_iterations',
                                                  num_iterations)
        self._gamma = check_value_positive('gamma', gamma)
        if constraint in ['l2', 'linf']:
            self._constraint = constraint
        else:
            msg = "constraint must be in ['l2', 'linf'], " \
                  "but got {}".format(constraint)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)
        self.queries = 0
        self.is_adv = True
        self.y_targets = None
        self.image_targets = None
        self.y_target = None
        self.image_target = None

    def _generate_one(self, sample):
        """
        Return a tensor that constructs adversarial examples for the given
        input.

        Args:
            sample (Tensor): Input samples.

        Returns:
            Tensor, generated adversarial examples.
        """
        shape = list(np.shape(sample))
        dim = int(np.prod(shape))

        # Set binary search threshold.
        if self._constraint == 'l2':
            theta = self._gamma / (np.sqrt(dim)*dim)
        else:
            theta = self._gamma / (dim*dim)

        wrap = self._hsja(sample, self.y_target, self.image_target, dim, theta)
        if wrap is None:
            self.is_adv = False
        else:
            self.is_adv = True
        return self.is_adv, wrap, self.queries

    def set_target_images(self, target_images):
        """
        Setting target images for target attack.

        Args:
            target_images (numpy.ndarray): Target images.
        """
        self.image_targets = check_numpy_param('target_images', target_images)

    def generate(self, inputs, labels):
        """
        Generate adversarial images in a for loop.

        Args:
            inputs (numpy.ndarray): Origin images.
            labels (numpy.ndarray): Target labels.

        Returns:
            - numpy.ndarray, bool values for each attack result.

            - numpy.ndarray, generated adversarial examples.

            - numpy.ndarray, query times for each sample.

        Examples:
            >>> generate([[0.1,0.2,0.2],[0.2,0.3,0.4]],[2,6])
        """
        if labels is not None:
            inputs, labels = check_pair_numpy_param('inputs', inputs,
                                                    'labels', labels)

        if not self._sparse:
            labels = np.argmax(labels, axis=1)
        x_adv = []
        is_advs = []
        queries_times = []

        if labels is not None:
            self.y_targets = labels

        for i, x_single in enumerate(inputs):
            self.queries = 0
            if self.image_targets is not None:
                self.image_target = self.image_targets[i]
            if self.y_targets is not None:
                self.y_target = self.y_targets[i]
            is_adv, adv_img, query_time = self._generate_one(x_single)
            x_adv.append(adv_img)
            is_advs.append(is_adv)
            queries_times.append(query_time)

        return np.asarray(is_advs), \
               np.asarray(x_adv), \
               np.asarray(queries_times)

    def _hsja(self, sample, target_label, target_image, dim, theta):
        """
        The main algorithm for HopSkipJumpAttack.

        Args:
            sample (numpy.ndarray): Input image. Without the batchsize
                dimension.
            target_label (int): Integer for targeted attack, None for
                nontargeted attack. Without the batchsize dimension.
            target_image (numpy.ndarray): An array with the same size as
                input sample, or None. Without the batchsize dimension.

        Returns:
            numpy.ndarray, perturbed images.
        """
        original_label = None
        # Original label for untargeted attack.
        if target_label is None:
            original_label = self._model.predict(sample)
            original_label = np.argmax(original_label)

        # Initialize perturbed image.
        # untarget attack
        if target_image is None:
            perturbed = self._initialize(sample, original_label, target_label)
            if perturbed is None:
                msg = 'Can not find an initial adversarial example'
                LOGGER.info(TAG, msg)
                return perturbed
        else:
            # Target attack
            perturbed = target_image

        # Project the initial perturbed image to the decision boundary.
        perturbed, dist_post_update = self._binary_search_batch(sample,
                                                                np.expand_dims(perturbed, 0),
                                                                original_label,
                                                                target_label,
                                                                theta)

        # Calculate the distance of perturbed image and original sample
        dist = self._compute_distance(perturbed, sample)
        for j in np.arange(self._num_iterations):
            current_iteration = j + 1

            # Select delta.
            delta = self._select_delta(dist_post_update, current_iteration, dim,
                                       theta)
            # Choose number of evaluations.
            num_evals = int(min([self._init_num_evals*np.sqrt(j + 1),
                                 self._max_num_evals]))

            # approximate gradient.
            gradf = self._approximate_gradient(perturbed, num_evals,
                                               original_label, target_label,
                                               delta, theta)
            if self._constraint == 'linf':
                update = np.sign(gradf)
            else:
                update = gradf

            # search step size.
            if self._stepsize_search == 'geometric_progression':
                # find step size.
                epsilon = self._geometric_progression_for_stepsize(
                    perturbed,
                    update,
                    dist,
                    current_iteration,
                    original_label,
                    target_label)
                # Update the sample.
                perturbed = _clip_image(perturbed + epsilon*update,
                                        self._clip_min, self._clip_max)

                # Binary search to return to the boundary.
                perturbed, dist_post_update = self._binary_search_batch(
                    sample,
                    perturbed[None],
                    original_label,
                    target_label,
                    theta)

            elif self._stepsize_search == 'grid_search':
                epsilons = np.logspace(-4, 0, num=20, endpoint=True)*dist
                epsilons_shape = [20] + len(np.shape(sample))*[1]
                perturbeds = perturbed + epsilons.reshape(
                    epsilons_shape)*update
                perturbeds = _clip_image(perturbeds, self._clip_min,
                                         self._clip_max)
                idx_perturbed = self._decision_function(perturbeds,
                                                        original_label,
                                                        target_label)

                if np.sum(idx_perturbed) > 0:
                    # Select the perturbation that yields the minimum distance
                    # after binary search.
                    perturbed, dist_post_update = self._binary_search_batch(
                        sample, perturbeds[idx_perturbed],
                        original_label, target_label, theta)

            # compute new distance.
            dist = self._compute_distance(perturbed, sample)

            LOGGER.debug(TAG,
                         'iteration: %d, %s distance %4f',
                         j + 1,
                         self._constraint, dist)

        perturbed = np.expand_dims(perturbed, 0)
        return perturbed

    def _decision_function(self, images, original_label, target_label):
        """
        Decision function returns 1 if the input sample is on the desired
        side of the boundary, and 0 otherwise.
        """
        images = _clip_image(images, self._clip_min, self._clip_max)
        prob = []
        self.queries += len(images)
        for i in range(0, len(images), self._batch_size):
            batch = images[i:i + self._batch_size]
            length = len(batch)
            prob_i = self._model.predict(batch)[:length]
            prob.append(prob_i)
        prob = np.concatenate(prob)
        if target_label is None:
            res = np.argmax(prob, axis=1) != original_label
        else:
            res = np.argmax(prob, axis=1) == target_label
        return res

    def _compute_distance(self, original_img, perturbation_img):
        """
        Compute the distance between original image and perturbation images.
        """
        if self._constraint == 'l2':
            distance = np.linalg.norm(original_img - perturbation_img)
        else:
            distance = np.max(abs(original_img - perturbation_img))
        return distance

    def _approximate_gradient(self, sample, num_evals, original_label,
                              target_label, delta, theta):
        """
        Gradient direction estimation.
        """
        # Generate random noise based on constraint.
        noise_shape = [num_evals] + list(np.shape(sample))
        if self._constraint == 'l2':
            random_noise = np.random.randn(*noise_shape)
        else:
            random_noise = np.random.uniform(low=-1, high=1, size=noise_shape)
        axis = tuple(range(1, 1 + len(np.shape(sample))))
        random_noise = random_noise / np.sqrt(
            np.sum(random_noise**2, axis=axis, keepdims=True))

        # perturbed images
        perturbed = sample + delta*random_noise
        perturbed = _clip_image(perturbed, self._clip_min, self._clip_max)
        random_noise = (perturbed - sample) / theta

        # Whether the perturbed images are on the desired side of the boundary.
        decisions = self._decision_function(perturbed, original_label,
                                            target_label)
        decision_shape = [len(decisions)] + [1]*len(np.shape(sample))
        # transform decisions value from 1, 0 to 1, -2
        re_decision = 2*np.array(decisions).astype(self._np_dtype).reshape(
            decision_shape) - 1.0

        if np.mean(re_decision) == 1.0:
            grad_direction = np.mean(random_noise, axis=0)
        elif np.mean(re_decision) == -1.0:
            grad_direction = - np.mean(random_noise, axis=0)
        else:
            re_decision = re_decision - np.mean(re_decision)
            grad_direction = np.mean(re_decision*random_noise, axis=0)

        # The gradient direction.
        grad_direction = grad_direction / (np.linalg.norm(grad_direction) + 1e-10)

        return grad_direction

    def _project(self, original_image, perturbed_images, alphas):
        """
        Projection input samples onto given l2 or linf balls.
        """
        alphas_shape = [len(alphas)] + [1]*len(np.shape(original_image))
        alphas = alphas.reshape(alphas_shape)
        if self._constraint == 'l2':
            projected = (1 - alphas)*original_image + alphas*perturbed_images
        else:
            projected = _clip_image(perturbed_images, original_image - alphas,
                                    original_image + alphas)

        return projected

    def _binary_search_batch(self, original_image, perturbed_images,
                             original_label, target_label, theta):
        """
        Binary search to approach the model decision boundary.
        """

        # Compute distance between perturbed image and original image.
        dists_post_update = np.array([self._compute_distance(original_image,
                                                             perturbed_image,)
                                      for perturbed_image in perturbed_images])

        # Get higher thresholds
        if self._constraint == 'l2':
            highs = np.ones(len(perturbed_images))
            thresholds = theta
        else:
            highs = dists_post_update
            thresholds = np.minimum(dists_post_update*theta, theta)

        # Get lower thresholds
        lows = np.zeros(len(perturbed_images))

        # Update thresholds.
        while np.max((highs - lows) / thresholds) > 1:
            mids = (highs + lows) / 2.0
            mid_images = self._project(original_image, perturbed_images, mids)
            decisions = self._decision_function(mid_images, original_label,
                                                target_label)
            lows = np.where(decisions == [0], mids, lows)
            highs = np.where(decisions == [1], mids, highs)

        out_images = self._project(original_image, perturbed_images, highs)

        # Select the best choice based on the distance of the output image.
        dists = np.array(
            [self._compute_distance(original_image, out_image) for out_image in
             out_images])
        idx = np.argmin(dists)

        dist = dists_post_update[idx]
        out_image = out_images[idx]
        return out_image, dist

    def _initialize(self, sample, original_label, target_label):
        """
        Implementation of BlendedUniformNoiseAttack
        """
        num_evals = 0

        while True:
            random_noise = np.random.uniform(self._clip_min, self._clip_max,
                                             size=np.shape(sample))
            success = self._decision_function(random_noise[None],
                                              original_label,
                                              target_label)
            if success:
                break
            num_evals += 1

            if num_evals > 1e3:
                return None

        # Binary search.
        low = 0.0
        high = 1.0
        while high - low > 0.001:
            mid = (high + low) / 2.0
            blended = (1 - mid)*sample + mid*random_noise
            success = self._decision_function(blended[None], original_label,
                                              target_label)
            if success:
                high = mid
            else:
                low = mid

        initialization = (1 - high)*sample + high*random_noise
        return initialization

    def _geometric_progression_for_stepsize(self, perturbed, update, dist,
                                            current_iteration, original_label,
                                            target_label):
        """
        Search for stepsize in the way of Geometric progression.
        Keep decreasing stepsize by half until reaching the desired side of
        the decision boundary.
        """
        epsilon = dist / np.sqrt(current_iteration)
        while True:
            updated = perturbed + epsilon*update
            success = self._decision_function(updated, original_label,
                                              target_label)
            if success:
                break
            epsilon = epsilon / 2.0

        return epsilon

    def _select_delta(self, dist_post_update, current_iteration, dim, theta):
        """
        Choose the delta based on the distance between the input sample
        and the perturbed sample.
        """
        if current_iteration == 1:
            delta = 0.1*(self._clip_max - self._clip_min)
        else:
            if self._constraint == 'l2':
                delta = np.sqrt(dim)*theta*dist_post_update
            else:
                delta = dim*theta*dist_post_update

        return delta
