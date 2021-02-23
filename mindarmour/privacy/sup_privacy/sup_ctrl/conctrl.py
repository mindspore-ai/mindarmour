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
               mask_times=1000, lr=0.10, sparse_end=0.90, sparse_start=0.0):
        """
        Args:
            networks (Cell): The training network.
            mask_layers (list): Description of the training network layers that need to be suppressed.
            policy (str): Training policy for suppress privacy training. Default: "local_train", means local training.
            end_epoch (int): The last epoch in suppress operations, 0<start_epoch<=end_epoch<=100. Default: 10.
            batch_num (int): The num of batch in an epoch, should be equal to num_samples/batch_size. Default: 20.
            start_epoch (int): The first epoch in suppress operations, 0<start_epoch<=end_epoch<=100. Default: 3.
            mask_times (int): The num of suppress operations. Default: 1000.
            lr (Union[float, int]): Learning rate, 0 < lr <= 0.5. Default: 0.10.
            sparse_end (Union[float, int]): The sparsity to reach, 0.0<=sparse_start<sparse_end<1.0. Default: 0.90.
            sparse_start (Union[float, int]): The sparsity to start, 0.0<=sparse_start<sparse_end<1.0. Default: 0.0.

        Returns:
            SuppressCtrl, class of Suppress Privavy Mechanism.

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
        sparse_end (Union[float, int]): The sparsity to reach.
        sparse_start (Union[float, int]): The sparsity to start.
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
        self.sparse_end = check_value_non_negative('sparse_end', sparse_end)
        self.sparse_start = check_value_non_negative('sparse_start', sparse_start)

        self.weight_lower_bound = 0.005  # all network weight will be larger than this value
        self.sparse_vibra = 0.02  # the sparsity may have certain range of variations
        self.sparse_valid_max_weight = 0.20  # if max network weight is less than this value, suppress operation stop temporarily
        self.add_noise_thd = 0.50  # if network weight is more than this value, noise is forced
        self.noise_volume = 0.01  # noise volume 0.01
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
            msg += "\n        mask_interval = (epoch_end-epoch_start+1)*batch_num, batch_num = samples/batch_size"
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

    def update_mask(self, networks, cur_step):
        """
        Update add mask arrays and multiply mask arrays of network layers.

        Args:
            networks (Cell): The training network.
            cur_step (int): Current epoch of the whole training process.
        """
        if self.sparse_end <= 0.0:
            return

        self.cur_sparse = self.sparse_end +\
                          (self.sparse_start - self.sparse_end)*\
                          math.pow((1.0 - (cur_step + 0.0 - self.mask_start_step) / self.mask_all_steps), 3)
        m = 0
        for layer in networks.get_parameters(expand=True):
            grad_idx = self.grad_idx_map[m]
            if grad_idx < 0:
                m = m + 1
                continue
            if self.grads_mask_list[grad_idx].mask_able:
                weight_array = layer.data.asnumpy()
                weight_avg = np.mean(weight_array)
                weight_array_flat = weight_array.flatten()
                weight_array_flat_abs = np.abs(weight_array_flat)
                weight_abs_avg = np.mean(weight_array_flat_abs)
                weight_array_flat_abs.sort()
                len_array = weight_array.size
                weight_abs_max = np.max(weight_array_flat_abs)
                if m == 0 and weight_abs_max < self.sparse_valid_max_weight:
                    msg = "give up this masking .."
                    LOGGER.info(TAG, msg)
                    return
                if self.grads_mask_list[grad_idx].min_num > 0:
                    sparse_weight_thd, _, actual_stop_pos = self.calc_sparse_thd(weight_array_flat_abs,
                                                                                 self.cur_sparse, grad_idx)
                else:
                    actual_stop_pos = int(len_array * self.cur_sparse)
                    sparse_weight_thd = weight_array_flat_abs[actual_stop_pos]

                self.update_mask_layer(weight_array_flat, sparse_weight_thd, actual_stop_pos, weight_abs_max, grad_idx)

                msg = "{} len={}, sparse={}, current sparse thd={}, max={}, avg={}, avg_abs={} \n".format(
                    layer.name, len_array, actual_stop_pos/len_array, sparse_weight_thd,
                    weight_abs_max, weight_avg, weight_abs_avg)
                LOGGER.info(TAG, msg)
            m = m + 1

    def update_mask_layer(self, weight_array_flat, sparse_weight_thd, sparse_stop_pos, weight_abs_max, layer_index):
        """
        Update add mask arrays and multiply mask arrays of one single layer.

        Args:
            weight_array (numpy.ndarray): The weight array of layer's parameters.
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
        for i in range(0, weight_array_flat.size):
            if abs(weight_array_flat[i]) <= sparse_weight_thd:
                if m < weight_array_flat.size - min_num and m < sparse_stop_pos:
                    # to mask
                    mul_mask_array_flat[i] = 0.0
                    add_mask_array_flat[i] = weight_array_flat[i] / self.lr
                    m = m + 1
                else:
                    # not mask
                    if weight_array_flat[i] > 0.0:
                        add_mask_array_flat[i] = (weight_array_flat[i] - self.weight_lower_bound) / self.lr
                    else:
                        add_mask_array_flat[i] = (weight_array_flat[i] + self.weight_lower_bound) / self.lr
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

    def calc_sparse_thd(self, array_flat, sparse_value, layer_index):
        """
        Calculate the suppression threshold of one weight array.

        Args:
            array_flat (numpy.ndarray): The flattened weight array.
            sparse_value (float): The target sparse value of weight array.

        Returns:
            - float, the sparse threshold of this array.

            - int, the number of weight elements to be suppressed.

            - int, the larger number of weight elements to be suppressed.
        """
        size = len(array_flat)
        sparse_max_thd = 1.0 - min(self.grads_mask_list[layer_index].min_num, size) / size
        pos = int(size*min(sparse_max_thd, sparse_value))
        thd = array_flat[pos]
        farther_stop_pos = int(size*min(sparse_max_thd, max(0, sparse_value + self.sparse_vibra / 2.0)))
        return thd, pos, farther_stop_pos

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
        sparse = sparse/full
        sparse_value_1 = sparse_value_1/full_conv1
        sparse_value_2 = sparse_value_2/full_conv2
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

        array_cur_conv1 = np.ones(np.shape([1]), dtype=np.float32)
        array_cur_conv2 = np.ones(np.shape([1]), dtype=np.float32)
        for layer in networks.get_parameters(expand=True):
            if "conv1.weight" in layer.name:
                array_cur_conv1 = layer.data.asnumpy()
            if "conv2.weight" in layer.name:
                array_cur_conv2 = layer.data.asnumpy()

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
        return sparse, sparse_value_1, sparse_value_2

    def calc_actual_sparse_for_fc1(self, networks):
        self.calc_actual_sparse_for_layer(networks, "fc1.weight")

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

        if array_cur is None:
            msg = "no such layer to calc sparse: {} ".format(layer_name)
            LOGGER.info(TAG, msg)
            return

        array_cur_flat = array_cur.flatten()

        for i in range(0, array_cur_flat.size):
            full += 1.0
            if abs(array_cur_flat[i]) <= self.base_ground_thd:
                sparse += 1.0

        sparse = sparse / full
        msg = "{} sparse fact={} ".format(layer_name, sparse)
        LOGGER.info(TAG, msg)

    def print_paras(self):
        msg = "paras: start_epoch:{}, end_epoch:{}, batch_num:{}, interval:{}, lr:{}, sparse_end:{}, sparse_start:{}" \
            .format(self.mask_start_epoch, self.mask_end_epoch, self.batch_num, self.mask_step_interval,
                    self.lr, self.sparse_end, self.sparse_start)
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
        is_lower_clip (bool): If true, the weights of this layer would be clipped to greater than an lower bound value.
            If False, the weights of this layer won't be clipped.
        min_num (int): The number of weights left that not be suppressed.
            If min_num is smaller than (parameter num*SupperssCtrl.sparse_end), min_num has not effect.
        upper_bound (Union[float, int]): max abs value of weight in this layer, default: 1.20.
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
        is_lower_clip (bool): If true, the weights of this layer would be clipped to greater than an lower bound value.
            If False, the weights of this layer won't be clipped.
        min_num (int): The number of weights left that not be suppressed.
            If min_num is smaller than (parameter num*SupperssCtrl.sparse_end), min_num has not effect.
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
        self.upper_bound = check_value_positive('upper_bound', upper_bound)

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
