# Copyright 2021 Huawei Technologies Co., Ltd
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
control function of suppress-based privacy.
"""
import math
import gc
import numpy as np

from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.nn import Cell

from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_int_positive, check_value_positive, \
    check_value_non_negative, check_param_type
LOGGER = LogUtil.get_instance()
TAG = 'Suppression training.'

class SuppressPrivacyFactory:
    """ Factory class of SuppressCtrl mechanisms"""
    def __init__(self):
        pass

    @staticmethod
    def create(networks, mask_layers, policy="local_train", end_epoch=10, batch_num=20, start_epoch=3,
               mask_times=1000, lr=0.05, sparse_end=0.90, sparse_start=0.0):
        """
        Args:
            networks (Cell): The training network.
                This networks parameter should be same as 'network' parameter of SuppressModel().
            mask_layers (list): Description of the training network layers that need to be suppressed.
            policy (str): Training policy for suppress privacy training. Default: "local_train", means local training.
            end_epoch (int): The last epoch in suppress operations, 0<start_epoch<=end_epoch<=100. Default: 10.
                This end_epoch parameter should be same as 'epoch' parameter of mindspore.train.model.train().
            batch_num (int): The num of batch in an epoch, should be equal to num_samples/batch_size. Default: 20.
            start_epoch (int): The first epoch in suppress operations, 0<start_epoch<=end_epoch<=100. Default: 3.
            mask_times (int): The num of suppress operations. Default: 1000.
            lr (Union[float, int]): Learning rate, should be unchanged during training. 0<lr<=0.50. Default: 0.05.
                This lr parameter should be same as 'learning_rate' parameter of mindspore.nn.SGD().
            sparse_end (float): The sparsity to reach, 0.0<=sparse_start<sparse_end<1.0. Default: 0.90.
            sparse_start (Union[float, int]): The sparsity to start, 0.0<=sparse_start<sparse_end<1.0. Default: 0.0.

        Returns:
            SuppressCtrl, class of Suppress Privavy Mechanism.

        Examples:
            >>> networks_l5 = LeNet5()
            >>> mask_layers = []
            >>> mask_layers.append(MaskLayerDes("conv1.weight", 0, False, True, 10))
            >>> suppress_ctrl_instance = SuppressPrivacyFactory().create(networks=networks_l5,
            >>>                                                 mask_layers=mask_layers,
            >>>                                                 policy="local_train",
            >>>                                                 end_epoch=10,
            >>>                                                 batch_num=(int)(10000/cfg.batch_size),
            >>>                                                 start_epoch=3,
            >>>                                                 mask_times=1000,
            >>>                                                 lr=lr,
            >>>                                                 sparse_end=0.90,
            >>>                                                 sparse_start=0.0)
            >>> net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
            >>> net_opt = nn.Momentum(params=networks_l5.trainable_params(), learning_rate=lr, momentum=0.0)
            >>> config_ck = CheckpointConfig(save_checkpoint_steps=(int)(samples/cfg.batch_size),
            >>>                              keep_checkpoint_max=10)
            >>> model_instance = SuppressModel(network=networks_l5,
            >>>                             loss_fn=net_loss,
            >>>                             optimizer=net_opt,
            >>>                             metrics={"Accuracy": Accuracy()})
            >>> model_instance.link_suppress_ctrl(suppress_ctrl_instance)
            >>> ds_train = generate_mnist_dataset("./MNIST_unzip/train",
            >>>                                 batch_size=cfg.batch_size, repeat_size=1, samples=samples)
            >>> ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet",
            >>>                             directory="./trained_ckpt_file/",
            >>>                             config=config_ck)
            >>> model_instance.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(), suppress_masker],
            >>>                 dataset_sink_mode=False)
        """
        check_param_type('policy', policy, str)
        if policy == "local_train":
            return SuppressCtrl(networks, mask_layers, end_epoch, batch_num, start_epoch, mask_times, lr,
                                sparse_end, sparse_start)
        msg = "Only local training is supported now, but got {}.".format(policy)
        LOGGER.error(TAG, msg)
        raise ValueError(msg)

class SuppressCtrl(Cell):
    """
    Args:
        networks (Cell): The training network.
        mask_layers (list): Description of those layers that need to be suppressed.
        end_epoch (int): The last epoch in suppress operations.
        batch_num (int): The num of grad operation in an epoch.
        start_epoch (int): The first epoch in suppress operations.
        mask_times (int): The num of suppress operations.
        lr (Union[float, int]): Learning rate.
        sparse_end (float): The sparsity to reach.
        sparse_start (Union[float, int]): The sparsity to start.

    Examples:
        >>> networks_l5 = LeNet5()
        >>> masklayers = []
        >>> masklayers.append(MaskLayerDes("conv1.weight", 0, False, True, 10))
        >>> suppress_ctrl_instance = SuppressPrivacyFactory().create(networks=networks_l5,
        >>>                                                 mask_layers=masklayers,
        >>>                                                 policy="local_train",
        >>>                                                 end_epoch=10,
        >>>                                                 batch_num=(int)(10000/cfg.batch_size),
        >>>                                                 start_epoch=3,
        >>>                                                 mask_times=1000,
        >>>                                                 lr=lr,
        >>>                                                 sparse_end=0.90,
        >>>                                                 sparse_start=0.0)
        >>> net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        >>> net_opt = nn.Momentum(params=networks_l5.trainable_params(), learning_rate=lr, momentum=0.0)
        >>> config_ck = CheckpointConfig(save_checkpoint_steps=(int)(samples/cfg.batch_size),
        >>>                              keep_checkpoint_max=10)
        >>> model_instance = SuppressModel(network=networks_l5,
        >>>                             loss_fn=net_loss,
        >>>                             optimizer=net_opt,
        >>>                             metrics={"Accuracy": Accuracy()})
        >>> model_instance.link_suppress_ctrl(suppress_ctrl_instance)
        >>> ds_train = generate_mnist_dataset("./MNIST_unzip/train",
        >>>                                 batch_size=cfg.batch_size, repeat_size=1, samples=samples)
        >>> ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet",
        >>>                             directory="./trained_ckpt_file/",
        >>>                             config=config_ck)
        >>> model_instance.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(), suppress_masker],
        >>>                 dataset_sink_mode=False)
    """
    def __init__(self, networks, mask_layers, end_epoch, batch_num, start_epoch, mask_times, lr,
                 sparse_end, sparse_start):
        super(SuppressCtrl, self).__init__()
        self.networks = check_param_type('networks', networks, Cell)
        self.mask_layers = check_param_type('mask_layers', mask_layers, list)
        self.mask_end_epoch = check_int_positive('end_epoch', end_epoch)
        self.batch_num = check_int_positive('batch_num', batch_num)
        self.mask_start_epoch = check_int_positive('start_epoch', start_epoch)
        self.mask_times = check_int_positive('mask_times', mask_times)
        self.lr = check_value_positive('lr', lr)
        self.sparse_end = check_param_type('sparse_end', sparse_end, float)
        self.sparse_start = check_value_non_negative('sparse_start', sparse_start)

        self.weight_lower_bound = 0.005  # all network weight will be larger than this value
        self.sparse_vibra = 0.02  # the sparsity may have certain range of variations
        self.sparse_valid_max_weight = 0.02 # if max network weight is less than this value, suppress operation stop temporarily
        self.add_noise_thd = 0.50  # if network weight is more than this value, noise is forced
        self.noise_volume = 0.1  # noise volume 0.1
        self.base_ground_thd = 0.0000001  # if network weight is less than this value, will be considered as 0
        self.model = None  # SuppressModel instance
        self.grads_mask_list = []  # list for Grad Mask Matrix tensor
        self.de_weight_mask_list = []  # list for weight Mask Matrix tensor
        self.to_do_mask = False  # the flag means suppress operation is toggled immediately
        self.mask_started = False  # the flag means suppress operation has been started
        self.mask_start_step = 0  # suppress operation is actually started at this step
        self.mask_prev_step = 0  # previous suppress operation is done at this step
        self.cur_sparse = 0.0  # current sparsity to which one suppress will get
        self.mask_all_steps = (end_epoch - start_epoch + 1)*batch_num  # the amount of step contained in all suppress operation
        self.mask_step_interval = self.mask_all_steps/mask_times  # the amount of step contaied in one suppress operation
        self.mask_initialized = False  # flag means the initialization is done
        self.grad_idx_map = []

        if self.lr > 0.5:
            msg = "learning rate should not be greater than 0.5, but got {}".format(self.lr)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

        if self.mask_start_epoch > self.mask_end_epoch:
            msg = "start_epoch should not be greater than end_epoch, but got start_epoch and end_epoch are: " \
                  "{}, {}".format(self.mask_start_epoch, self.mask_end_epoch)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

        if self.mask_end_epoch > 100:
            msg = "The end_epoch should be smaller than 100, but got {}".format(self.mask_end_epoch)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

        if self.mask_step_interval <= 0:
            msg = "step_interval should be greater than 0, but got {}".format(self.mask_step_interval)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

        if self.mask_step_interval <= 10 or self.mask_step_interval >= 20:
            msg = "mask_interval should be greater than 10, smaller than 20, but got {}".format(self.mask_step_interval)
            msg += "\n Precision of trained model may be poor !!! "
            msg += "\n please modify epoch_start, epoch_end and batch_num !"
            msg += "\n mask_interval = (epoch_end-epoch_start+1)*batch_num/mask_times, batch_num = samples/batch_size"
            LOGGER.info(TAG, msg)

        if self.sparse_end >= 1.00 or self.sparse_end <= 0:
            msg = "sparse_end should be in range (0, 1), but got {}".format(self.sparse_end)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

        if self.sparse_start >= self.sparse_end:
            msg = "sparse_start should be smaller than sparse_end, but got sparse_start and sparse_end are: " \
                  "{}, {}".format(self.sparse_start, self.sparse_end)
            LOGGER.error(TAG, msg)
            raise ValueError(msg)

        if mask_layers is not None:
            mask_layer_id = 0
            for one_mask_layer in mask_layers:
                if not isinstance(one_mask_layer, MaskLayerDes):
                    msg = "mask_layers should be a list of MaskLayerDes, but got a {}".format(type(one_mask_layer))
                    LOGGER.error(TAG, msg)
                    raise ValueError(msg)
                layer_name = one_mask_layer.layer_name
                mask_layer_id2 = 0
                for one_mask_layer_2 in mask_layers:
                    if mask_layer_id != mask_layer_id2 and layer_name == one_mask_layer_2.layer_name:
                        msg = "Mask layer name should be unique, but got duplicate name: {} in mask_layer {} and {}".\
                            format(layer_name, mask_layer_id, mask_layer_id2)
                        LOGGER.error(TAG, msg)
                        raise ValueError(msg)
                    if mask_layer_id != mask_layer_id2 and one_mask_layer.grad_idx == one_mask_layer_2.grad_idx:
                        msg = "Grad_idx should be unique, but got duplicate idx: {} in mask_layer {} and {}".\
                            format(layer_name, one_mask_layer_2.layer_name, one_mask_layer.grad_idx)
                        LOGGER.error(TAG, msg)
                        raise ValueError(msg)
                    mask_layer_id2 = mask_layer_id2 + 1
                mask_layer_id = mask_layer_id + 1

        if networks is not None:
            for layer in networks.get_parameters(expand=True):
                shape = np.shape([1])
                mul_mask_array = np.ones(shape, dtype=np.float32)
                grad_mask_cell = GradMaskInCell(mul_mask_array, False, False, -1)
                grad_mask_cell.mask_able = False
                self.grads_mask_list.append(grad_mask_cell)

                add_mask_array = np.zeros(shape, dtype=np.float32)
                de_weight_cell = DeWeightInCell(add_mask_array)
                de_weight_cell.mask_able = False
                self.de_weight_mask_list.append(de_weight_cell)

                self.grad_idx_map.append(-1)

            m = 0
            for layer in networks.get_parameters(expand=True):
                one_mask_layer = None
                if mask_layers is not None:
                    one_mask_layer = get_one_mask_layer(mask_layers, layer.name)
                if one_mask_layer is not None and not one_mask_layer.inited:
                    one_mask_layer.inited = True
                    shape = P.Shape()(layer)
                    mul_mask_array = np.ones(shape, dtype=np.float32)
                    grad_mask_cell = GradMaskInCell(mul_mask_array,
                                                    one_mask_layer.is_add_noise,
                                                    one_mask_layer.is_lower_clip,
                                                    one_mask_layer.min_num,
                                                    one_mask_layer.upper_bound)
                    grad_mask_cell.mask_able = True
                    self.grads_mask_list[one_mask_layer.grad_idx] = grad_mask_cell

                    add_mask_array = np.zeros(shape, dtype=np.float32)
                    de_weight_cell = DeWeightInCell(add_mask_array)
                    de_weight_cell.mask_able = True
                    self.de_weight_mask_list[one_mask_layer.grad_idx] = de_weight_cell
                    self.grad_idx_map[m] = one_mask_layer.grad_idx
                    msg = "do mask {}, {}, {}".format(m, one_mask_layer.layer_name, one_mask_layer.grad_idx)
                    LOGGER.info(TAG, msg)
                elif one_mask_layer is not None and one_mask_layer.inited:
                    msg = "repeated match masked setting {}=>{}.".format(one_mask_layer.layer_name, layer.name)
                    LOGGER.error(TAG, msg)
                    raise ValueError(msg)
                m += 1
            self.mask_initialized = True
            msg = "init SuppressCtrl by networks"
            LOGGER.info(TAG, msg)
        msg = "complete init mask for lenet5.step_interval: {}".format(self.mask_step_interval)
        LOGGER.info(TAG, msg)

        for one_mask_layer in mask_layers:
            if not one_mask_layer.inited:
                msg = "can't match this mask layer: {} ".format(one_mask_layer.layer_name)
                LOGGER.error(TAG, msg)
                raise ValueError(msg)
        msg = "\nThis networks parameter should be same as 'network' parameter of SuppressModel()"
        msg += "\nThis lr parameter should be same as 'learning_rate' parameter of mindspore.nn.SGD()\n"
        msg += "\nThis end_epoch parameter should be same as 'epoch' parameter of mindspore.train.model.train()\n"
        msg += "\nsup_privacy only support SGD optimizer"
        LOGGER.warn(TAG, msg)

    def update_status(self, cur_epoch, cur_step, cur_step_in_epoch):
        """
        Update the suppress operation status.

        Args:
            cur_epoch (int): Current epoch of the whole training process.
            cur_step (int): Current step of the whole training process.
            cur_step_in_epoch (int): Current step of the current epoch.
        """
        if not self.mask_initialized:
            self.mask_started = False
        elif (self.mask_start_epoch <= cur_epoch <= self.mask_end_epoch) or self.mask_started:
            if not self.mask_started:
                self.mask_started = True
                self.mask_start_step = cur_step
            if cur_step >= (self.mask_prev_step + self.mask_step_interval):
                self.mask_prev_step = cur_step
                self.to_do_mask = True
            # execute the last suppression operation
            elif cur_epoch == self.mask_end_epoch and cur_step_in_epoch == self.batch_num-2:
                self.mask_prev_step = cur_step
                self.to_do_mask = True
            else:
                self.to_do_mask = False
        else:
            self.to_do_mask = False
            self.mask_started = False

    def update_mask(self, networks, cur_step, target_sparse=0.0):
        """
        Update add mask arrays and multiply mask arrays of network layers.

        Args:
            networks (Cell): The training network.
            cur_step (int): Current epoch of the whole training process.
            target_sparse(float): The sparsity to reach. Default: 0.0.
        """
        if self.sparse_end <= 0.0:
            return

        last_sparse = self.cur_sparse
        if target_sparse > 0.0:
            self.cur_sparse = target_sparse
        else:
            self.cur_sparse = self.sparse_end + \
                              (self.sparse_start - self.sparse_end) * \
                              math.pow((1.0 - (cur_step + 0.0 - self.mask_start_step) / self.mask_all_steps), 3)
        self.cur_sparse = max(self.sparse_start, max(last_sparse, min(self.cur_sparse, self.sparse_end)))
        m = 0
        for layer in networks.get_parameters(expand=True):
            grad_idx = self.grad_idx_map[m]
            if grad_idx < 0:
                m = m + 1
                continue
            if self.grads_mask_list[grad_idx].mask_able:
                len_array = self.grads_mask_list[grad_idx].para_num
                min_num = self.grads_mask_list[grad_idx].min_num
                sparse_min_thd = 1.0 - min(min_num, len_array) / len_array
                actual_stop_pos = int(len_array * min(sparse_min_thd, self.cur_sparse))

                grad_mask_cell = self.grads_mask_list[grad_idx]
                last_sparse_pos = grad_mask_cell.sparse_pos_list[-1]
                if actual_stop_pos <= 0 or \
                    (actual_stop_pos < last_sparse_pos + grad_mask_cell.part_num and \
                     grad_mask_cell.is_approximity and m > 0):
                    sparse_weight_thd = 0
                    msg = "{} len={}, sparse={}, current sparse thd={}, [idle] \n" \
                        .format(layer.name, len_array, actual_stop_pos / len_array, sparse_weight_thd)
                    LOGGER.info(TAG, msg)
                    m = m + 1
                    continue

                weight_array = layer.data.asnumpy()
                weight_avg = np.mean(weight_array)
                weight_array_flat = weight_array.flatten()
                weight_array_flat_abs = np.abs(weight_array_flat)
                weight_abs_avg = np.mean(weight_array_flat_abs)
                weight_abs_max = np.max(weight_array_flat_abs)
                weight_abs_min = np.min(weight_array_flat_abs)

                if m == 0 and weight_abs_max < self.sparse_valid_max_weight:
                    msg = "layer 0 weight_abs_max = {}, give up this masking ... ".format(weight_abs_max)
                    LOGGER.info(TAG, msg)
                    del weight_array_flat_abs
                    del weight_array_flat
                    del weight_array
                    gc.collect()
                    return

                if grad_mask_cell.is_approximity and m > 0:
                    sparse_weight_thd = self.update_mask_layer_approximity(weight_array_flat, weight_array_flat_abs,
                                                                           actual_stop_pos, grad_idx)
                else:
                    partition = np.partition(weight_array_flat_abs, actual_stop_pos - 1)
                    sparse_weight_thd = partition[actual_stop_pos - 1]
                    self.update_mask_layer(weight_array_flat, sparse_weight_thd, actual_stop_pos,
                                           weight_abs_max, grad_idx)
                    del partition

                msg = "{} len={}, sparse={}, current sparse thd={}, max={}, min={}, avg={}, avg_abs={} \n".format(
                    layer.name, len_array, actual_stop_pos/len_array, sparse_weight_thd,
                    weight_abs_max, weight_abs_min, weight_avg, weight_abs_avg)
                LOGGER.info(TAG, msg)
                del weight_array_flat_abs
                del weight_array_flat
                del weight_array
                gc.collect()
            m = m + 1

    def update_mask_layer(self, weight_array_flat, sparse_weight_thd, sparse_stop_pos, weight_abs_max, layer_index):
        """
        Update add mask arrays and multiply mask arrays of one single layer.

        Args:
            weight_array_flat (numpy.ndarray): The weight array of layer's parameters.
            sparse_weight_thd (float): The weight threshold of sparse operation.
            sparse_stop_pos (int): The maximum number of elements to be suppressed.
            weight_abs_max (float): The maximum absolute value of weights.
            layer_index (int): The index of target layer.
        """
        grad_mask_cell = self.grads_mask_list[layer_index]
        mul_mask_array_flat = grad_mask_cell.mul_mask_array_flat
        de_weight_cell = self.de_weight_mask_list[layer_index]
        add_mask_array_flat = de_weight_cell.add_mask_array_flat
        min_num = grad_mask_cell.min_num
        is_add_noise = grad_mask_cell.is_add_noise
        is_lower_clip = grad_mask_cell.is_lower_clip
        upper_bound = grad_mask_cell.upper_bound

        if not self.grads_mask_list[layer_index].mask_able:
            return
        m = 0
        n = 0
        p = 0
        q = 0
        # add noise on weights if not masking or clipping.
        weight_noise_bound = min(self.add_noise_thd, max(self.noise_volume*10, weight_abs_max*0.75))
        size = self.grads_mask_list[layer_index].para_num
        for i in range(0, size):
            if mul_mask_array_flat[i] <= 0.0:
                add_mask_array_flat[i] = weight_array_flat[i] / self.lr
                m = m + 1
            elif abs(weight_array_flat[i]) <= sparse_weight_thd:
                if m < size - min_num and m < sparse_stop_pos:
                    # to mask
                    mul_mask_array_flat[i] = 0.0
                    add_mask_array_flat[i] = weight_array_flat[i] / self.lr
                    m = m + 1
                else:
                    # not mask
                    if weight_array_flat[i] > 0.0:
                        add_mask_array_flat[i] = (weight_array_flat[i] \
                                                  - min(self.weight_lower_bound, sparse_weight_thd)) / self.lr
                    else:
                        add_mask_array_flat[i] = (weight_array_flat[i]
                                                  + min(self.weight_lower_bound, sparse_weight_thd)) / self.lr
                    p = p + 1
            elif is_lower_clip and abs(weight_array_flat[i]) <= \
                    self.weight_lower_bound and sparse_weight_thd > self.weight_lower_bound*0.5:
                # not mask
                mul_mask_array_flat[i] = 1.0
                if weight_array_flat[i] > 0.0:
                    add_mask_array_flat[i] = (weight_array_flat[i] - self.weight_lower_bound) / self.lr
                else:
                    add_mask_array_flat[i] = (weight_array_flat[i] + self.weight_lower_bound) / self.lr
                p = p + 1
            elif abs(weight_array_flat[i]) > upper_bound:
                mul_mask_array_flat[i] = 1.0
                if weight_array_flat[i] > 0.0:
                    add_mask_array_flat[i] = (weight_array_flat[i] - upper_bound) / self.lr
                else:
                    add_mask_array_flat[i] = (weight_array_flat[i] + upper_bound) / self.lr
                n = n + 1
            else:
                # not mask
                mul_mask_array_flat[i] = 1.0
                if is_add_noise and abs(weight_array_flat[i]) > weight_noise_bound > 0.0:
                    # add noise
                    add_mask_array_flat[i] = np.random.uniform(-self.noise_volume, self.noise_volume) / self.lr
                    q = q + 1
                else:
                    add_mask_array_flat[i] = 0.0

        grad_mask_cell.update()
        de_weight_cell.update()
        msg = "Dimension of mask tensor is {}D, which located in the {}-th layer of the network. \n The number of " \
              "suppressed elements, max-clip elements, min-clip elements and noised elements are {}, {}, {}, {}"\
                  .format(len(grad_mask_cell.mul_mask_array_shape), layer_index, m, n, p, q)
        LOGGER.info(TAG, msg)
        grad_mask_cell.sparse_pos_list.append(m)

    def update_mask_layer_approximity(self, weight_array_flat, weight_array_flat_abs, actual_stop_pos, layer_index):
        """
        Update add mask arrays and multiply mask arrays of one single layer with many parameter.
        Disable clipping lower, clipping, adding noise operation

        Args:
            weight_array_flat (numpy.ndarray): The weight array of layer's parameters.
            weight_array_flat_abs (numpy.ndarray): The abs weight array of layer's parameters.
            actual_stop_pos (int): The actually para num should be suppressed.
            layer_index (int): The index of target layer.
        """
        grad_mask_cell = self.grads_mask_list[layer_index]
        mul_mask_array_flat = grad_mask_cell.mul_mask_array_flat
        de_weight_cell = self.de_weight_mask_list[layer_index]
        add_mask_array_flat = de_weight_cell.add_mask_array_flat

        part_size = grad_mask_cell.part_size
        part_num = grad_mask_cell.part_num
        para_num = grad_mask_cell.para_num
        init_batch_suppress = False

        if not self.grads_mask_list[layer_index].mask_able:
            return 0.0
        real_part_num = 0
        sparse_thd = 0.0
        last_sparse_pos = grad_mask_cell.sparse_pos_list[-1]
        split_k_num = max(0, int((actual_stop_pos - last_sparse_pos) / part_num))
        if last_sparse_pos <= 0:
            init_batch_suppress = True
        for i in range(0, part_num):
            if split_k_num <= 0:
                break
            array_row_mul_mask = mul_mask_array_flat[i * part_size : (i + 1) * part_size]
            array_row_flat_abs = weight_array_flat_abs[i * part_size : (i + 1) * part_size]
            if not init_batch_suppress:
                array_row_flat_abs_masked = np.where(array_row_mul_mask <= 0.0, -1.0, array_row_flat_abs)
                set_abs = set(array_row_flat_abs_masked)
                set_abs.remove(-1.0)
                list2 = list(set_abs)
                val_array_align = np.array(list2)
                del array_row_flat_abs_masked
                del set_abs
                del list2
            else:
                val_array_align = array_row_flat_abs

            real_split_k_num = min(split_k_num, len(val_array_align) - 1)
            if real_split_k_num <= 0:
                del array_row_flat_abs
                del array_row_mul_mask
                del val_array_align
                continue

            partition = np.partition(val_array_align, real_split_k_num - 1)
            sparse_k_thd = partition[real_split_k_num - 1]
            if sparse_k_thd > 0 or init_batch_suppress:
                real_part_num = real_part_num + 1
                sparse_thd = sparse_thd + sparse_k_thd
            del array_row_flat_abs
            del array_row_mul_mask
            del val_array_align
            del partition

        if real_part_num > 0:
            sparse_thd = sparse_thd / real_part_num
            new_mul_mask_array_flat = np.where(weight_array_flat_abs <= sparse_thd, 0.0, 1.0)
            grad_mask_cell.mul_mask_array_flat = new_mul_mask_array_flat
            new_add_mask_array_flat = np.where(new_mul_mask_array_flat <= 0.0, weight_array_flat / self.lr, 0.0)
            de_weight_cell.add_mask_array_flat = new_add_mask_array_flat
            grad_mask_cell.update()
            de_weight_cell.update()
            del mul_mask_array_flat
            del add_mask_array_flat
            gc.collect()
            real_suppress_num = para_num - int(np.sum(grad_mask_cell.mul_mask_array_flat))
            grad_mask_cell.sparse_pos_list.append(real_suppress_num)
        else:
            real_suppress_num = 0

        msg = "Dimension of mask tensor is {}D, which located in the {}-th layer of the network. " \
              "\n The ideal number of suppressed elements is {}/{}/{}, real suppress elements is {}" \
            .format(len(grad_mask_cell.mul_mask_array_shape), layer_index,
                    split_k_num, (actual_stop_pos - last_sparse_pos), actual_stop_pos, real_suppress_num)
        LOGGER.info(TAG, msg)
        if init_batch_suppress:
            init_sparse_actual = real_suppress_num/para_num
            print("init batch suppresss, actual sparse = {}".format(init_sparse_actual))

        gc.collect()
        return sparse_thd

    def reset_zeros(self):
        """
        Set add mask arrays to be zero.
        """
        for de_weight_cell in self.de_weight_mask_list:
            de_weight_cell.reset_zeros()

    def calc_theoretical_sparse_for_conv(self):
        """
        Compute actually sparsity of mask matrix for conv1 layer and conv2 layer.
        """
        array_mul_mask_flat_conv1 = self.grads_mask_list[0].mul_mask_array_flat
        array_mul_mask_flat_conv2 = self.grads_mask_list[1].mul_mask_array_flat
        sparse = 0.0
        sparse_value_1 = 0.0
        sparse_value_2 = 0.0
        full = 0.0
        full_conv1 = 0.0
        full_conv2 = 0.0
        for i in range(0, array_mul_mask_flat_conv1.size):
            full += 1.0
            full_conv1 += 1.0
            if array_mul_mask_flat_conv1[i] <= 0.0:
                sparse += 1.0
                sparse_value_1 += 1.0
        for i in range(0, array_mul_mask_flat_conv2.size):
            full = full + 1.0
            full_conv2 = full_conv2 + 1.0
            if array_mul_mask_flat_conv2[i] <= 0.0:
                sparse = sparse + 1.0
                sparse_value_2 += 1.0
        sparse = sparse / full
        sparse_value_1 = sparse_value_1 / full_conv1
        sparse_value_2 = sparse_value_2 / full_conv2
        msg = "conv sparse mask={}, sparse_1={}, sparse_2={}".format(sparse, sparse_value_1, sparse_value_2)
        LOGGER.info(TAG, msg)
        return sparse, sparse_value_1, sparse_value_2

    def calc_actual_sparse_for_conv(self, networks):
        """
        Compute actually sparsity of network for conv1 layer and conv2 layer.

        Args:
            networks (Cell): The training network.
        """
        sparse = 0.0
        sparse_value_1 = 0.0
        sparse_value_2 = 0.0
        full = 0.0
        full_conv1 = 0.0
        full_conv2 = 0.0

        conv1_matched = False
        conv2_matched = False

        array_cur_conv1 = np.ones(np.shape([1]), dtype=np.float32)
        array_cur_conv2 = np.ones(np.shape([1]), dtype=np.float32)
        for layer in networks.get_parameters(expand=True):
            if not conv1_matched and \
                    ("networks.conv1.weight" in layer.name or "networks.layers.0.weight" in layer.name):
                # lenet5/res50 vgg16
                array_cur_conv1 = layer.data.asnumpy()
                print("calc_actual_sparse, match conv1: {}".format(layer.name))
                conv1_matched = True
            if not conv2_matched and \
                    ("networks.conv2.weight" in layer.name or "networks.layers.3.weight" in layer.name \
                     or "networks.layer1.0.conv1.weight" in layer.name):  # res50
                array_cur_conv2 = layer.data.asnumpy()
                print("calc_actual_sparse, match conv2: {}".format(layer.name))
                conv2_matched = True

        array_mul_mask_flat_conv1 = array_cur_conv1.flatten()
        array_mul_mask_flat_conv2 = array_cur_conv2.flatten()

        for i in range(0, array_mul_mask_flat_conv1.size):
            full += 1.0
            full_conv1 += 1.0
            if abs(array_mul_mask_flat_conv1[i]) <= self.base_ground_thd:
                sparse += 1.0
                sparse_value_1 += 1.0

        for i in range(0, array_mul_mask_flat_conv2.size):
            full = full + 1.0
            full_conv2 = full_conv2 + 1.0
            if abs(array_mul_mask_flat_conv2[i]) <= self.base_ground_thd:
                sparse = sparse + 1.0
                sparse_value_2 += 1.0

        sparse = sparse / full
        sparse_value_1 = sparse_value_1 / full_conv1
        sparse_value_2 = sparse_value_2 / full_conv2
        msg = "conv sparse fact={}, sparse_1={}, sparse_2={}".format(sparse, sparse_value_1, sparse_value_2)
        LOGGER.info(TAG, msg)
        del array_mul_mask_flat_conv1
        del array_mul_mask_flat_conv2
        del array_cur_conv1
        del array_cur_conv2
        gc.collect()
        return sparse, sparse_value_1, sparse_value_2

    def calc_actual_sparse_for_fc1(self, networks):
        return self.calc_actual_sparse_for_layer(networks, "fc1.weight")

    def calc_actual_sparse_for_layer(self, networks, layer_name):
        """
        Compute actually sparsity of one network layer

        Args:
            networks (Cell): The training network.
            layer_name (str): The name of target layer.
        """
        check_param_type('networks', networks, Cell)
        check_param_type('layer_name', layer_name, str)

        sparse = 0.0
        full = 0.0

        array_cur = None
        for layer in networks.get_parameters(expand=True):
            if layer_name in layer.name:
                array_cur = layer.data.asnumpy()
                break

        if array_cur is None:
            msg = "no such layer to calc sparse: {} ".format(layer_name)
            LOGGER.info(TAG, msg)
            return 0.0

        array_cur_flat = array_cur.flatten()

        for i in range(0, array_cur_flat.size):
            full += 1.0
            if abs(array_cur_flat[i]) <= self.base_ground_thd:
                sparse += 1.0

        sparse = sparse / full
        msg = "{} sparse fact={} ".format(layer_name, sparse)
        LOGGER.info(TAG, msg)
        del array_cur_flat
        del array_cur
        gc.collect()
        return sparse

    def print_paras(self):
        """
        Show parameters info
        """
        msg = "paras: start_epoch:{}, end_epoch:{}, batch_num:{}, interval:{}, lr:{}, sparse_end:{}, sparse_start:{}" \
            .format(self.mask_start_epoch, self.mask_end_epoch, self.batch_num, self.mask_step_interval,
                    self.lr, self.sparse_end, self.sparse_start)
        LOGGER.info(TAG, msg)
        msg = "\nThis networks parameter should be same as 'network' parameter of SuppressModel()"
        msg = "\nThis lr parameter should be same as 'learning_rate' parameter of mindspore.nn.SGD()"
        msg += "\nThis end_epoch parameter should be same as 'epoch' parameter of mindspore.train.model.train()"
        msg += "\nsup_privacy only support SGD optimizer"
        LOGGER.info(TAG, msg)

def get_one_mask_layer(mask_layers, layer_name):
    """
    Returns the layer definitions that need to be suppressed.

    Args:
        mask_layers (list): The layers that need to be suppressed.
        layer_name (str): The name of target layer.

    Returns:
        Union[MaskLayerDes, None], the layer definitions that need to be suppressed.
    """
    for each_mask_layer in mask_layers:
        if each_mask_layer.layer_name in layer_name and not each_mask_layer.inited:
            return each_mask_layer
    return None

class MaskLayerDes:
    """
    Describe the layer that need to be suppressed.

    Args:
        layer_name (str): Layer name, get the name of one layer as following:

            .. code-block::

                for layer in networks.get_parameters(expand=True):
                    if layer.name == "conv": ...

        grad_idx (int): Grad layer index, get mask layer's index in grad tuple.You can refer to the construct function
            of TrainOneStepCell in mindarmour/privacy/sup_privacy/train/model.py to get the index of some specified
            grad layers (print in PYNATIVE_MODE).
        is_add_noise (bool): If True, the weight of this layer can add noise.
            If False, the weight of this layer can not add noise.
            If parameter num is greater than 100000, is_add_noise has no effect.
        is_lower_clip (bool): If True, the weights of this layer would be clipped to greater than an lower bound value.
            If False, the weights of this layer won't be clipped.
            If parameter num is greater than 100000, is_lower_clip has no effect.
        min_num (int): The number of weights left that not be suppressed.
            If min_num is smaller than (parameter num*SupperssCtrl.sparse_end), min_num has not effect.
        upper_bound (Union[float, int]): max abs value of weight in this layer, default: 1.20.
            If parameter num is greater than 100000, upper_bound has not effect.

    Examples:
        >>> masklayers = []
        >>> masklayers.append(MaskLayerDes("conv1.weight", 0, False, True, 10))
    """
    def __init__(self, layer_name, grad_idx, is_add_noise, is_lower_clip, min_num, upper_bound=1.20):
        self.layer_name = check_param_type('layer_name', layer_name, str)
        check_param_type('grad_idx', grad_idx, int)
        self.grad_idx = check_value_non_negative('grad_idx', grad_idx)
        self.is_add_noise = check_param_type('is_add_noise', is_add_noise, bool)
        self.is_lower_clip = check_param_type('is_lower_clip', is_lower_clip, bool)
        self.min_num = check_param_type('min_num', min_num, int)
        self.upper_bound = check_value_positive('upper_bound', upper_bound)
        self.inited = False

class GradMaskInCell(Cell):
    """
    Define the mask matrix for gradients masking.

    Args:
        array (numpy.ndarray): The mask array.
        is_add_noise (bool): If True, the weight of this layer can add noise.
            If False, the weight of this layer can not add noise.
        is_lower_clip (bool): If True, the weights of this layer would be clipped to greater than an lower bound value.
            If False, the weights of this layer won't be clipped.
        min_num (int): The number of weights left that not be suppressed.
            If min_num is smaller than (parameter num*SupperssCtrl.sparse_end), min_num has no effect.
        upper_bound ([float, int]): max abs value of weight in this layer, default: 1.20.
    """
    def __init__(self, array, is_add_noise, is_lower_clip, min_num, upper_bound=1.20):
        super(GradMaskInCell, self).__init__()
        self.mul_mask_array_shape = array.shape
        mul_mask_array = array.copy()
        self.mul_mask_array_flat = mul_mask_array.flatten()
        self.mul_mask_tensor = Tensor(array, mstype.float32)
        self.mask_able = False
        self.is_add_noise = is_add_noise
        self.is_lower_clip = is_lower_clip
        self.min_num = min_num
        self.upper_bound = max(0.10, check_value_positive('upper_bound', upper_bound))

        self.para_num = array.size
        self.is_approximity = False
        self.sparse_pos_list = [0]
        self.part_num = 1
        self.part_size = self.para_num
        self.part_num_max = 16
        self.para_many_num = 10000
        self.para_huge_num = 10*10000*10000

        if self.para_num > self.para_many_num:
            self.is_approximity = True
            self.is_add_noise = False
            self.is_lower_clip = False

            ratio = 2
            if self.part_size > self.para_huge_num:
                while self.part_size % ratio == 0 and self.part_size > self.para_huge_num \
                        and self.part_num < self.part_num_max:
                    self.part_num = self.part_num * ratio
                    self.part_size = int(self.part_size / ratio)
            msg = "this layer has {} para, disable the operation of clipping lower, clipping upper_bound, " \
                  "adding noise. \n part_num={}, part_size={}" \
                .format(self.para_num, self.part_num, self.part_size)
            LOGGER.info(TAG, msg)

    def construct(self):
        """
        Return the mask matrix for optimization.
        """
        return self.mask_able, self.mul_mask_tensor

    def update(self):
        """
        Update the mask tensor.
        """
        self.mul_mask_tensor = Tensor(self.mul_mask_array_flat.reshape(self.mul_mask_array_shape), mstype.float32)

class DeWeightInCell(Cell):
    """
    Define the mask matrix for de-weight masking.

    Args:
        array (numpy.ndarray): The mask array.
    """
    def __init__(self, array):
        super(DeWeightInCell, self).__init__()
        self.add_mask_array_shape = array.shape
        add_mask_array = array.copy()
        self.add_mask_array_flat = add_mask_array.flatten()
        self.add_mask_tensor = Tensor(array, mstype.float32)
        self.mask_able = False
        self.zero_mask_tensor = Tensor(np.zeros(array.shape, np.float32), mstype.float32)
        self.just_update = -1.0

    def construct(self):
        """
        Return the mask matrix for optimization.
        """
        if self.just_update > 0.0:
            return self.mask_able, self.add_mask_tensor
        return self.mask_able, self.zero_mask_tensor

    def update(self):
        """
        Update the mask tensor.
        """
        self.just_update = 1.0
        self.add_mask_tensor = Tensor(self.add_mask_array_flat.reshape(self.add_mask_array_shape), mstype.float32)

    def reset_zeros(self):
        """
        Make the de-weight operation expired.
        """
        self.just_update = -1.0
