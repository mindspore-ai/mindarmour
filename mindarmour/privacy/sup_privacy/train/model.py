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
suppress-basd privacy model.
"""
from easydict import EasyDict as edict

from mindspore.train.model import Model
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore.train.amp import _config_level
from mindspore.common import dtype as mstype
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.parallel._utils import _get_parallel_mode
from mindspore.train.model import ParallelMode
from mindspore.train.amp import _do_keep_batchnorm_fp32
from mindspore.train.amp import _add_loss_network
from mindspore import nn
from mindspore import context
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.parallel._utils import _get_gradients_mean
from mindspore.parallel._utils import _get_device_num
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.nn import Cell
from mindspore.nn.optim import SGD
from mindarmour.utils._check_param import check_param_type
from mindarmour.utils.logger import LogUtil
from mindarmour.privacy.sup_privacy.sup_ctrl.conctrl import SuppressCtrl

LOGGER = LogUtil.get_instance()
TAG = 'Mask model'

GRADIENT_CLIP_TYPE = 1
_grad_scale = C.MultitypeFuncGraph("grad_scale")
_reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    """ grad scaling """
    return grad*F.cast(_reciprocal(scale), F.dtype(grad))


class SuppressModel(Model):
    """
    This class is overload mindspore.train.model.Model.

    Args:
        network (Cell): The training network.
        loss_fn (Cell): Computes softmax cross entropy between logits and labels.
        optimizer (Optimizer): optimizer instance.
        kwargs: Keyword parameters used for creating a suppress model.

    Examples:
        >>> networks_l5 = LeNet5()
        >>> mask_layers = []
        >>> mask_layers.append(MaskLayerDes("conv1.weight", 0, False, True, 10))
        >>> suppress_ctrl_instance = SuppressPrivacyFactory().create(networks=networks_l5,
        >>>                                                     mask_layers=mask_layers,
        >>>                                                     policy="local_train",
        >>>                                                     end_epoch=10,
        >>>                                                     batch_num=(int)(10000/cfg.batch_size),
        >>>                                                     start_epoch=3,
        >>>                                                     mask_times=1000,
        >>>                                                     lr=lr,
        >>>                                                     sparse_end=0.90,
        >>>                                                     sparse_start=0.0)
        >>> net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        >>> net_opt = nn.Momentum(params=networks_l5.trainable_params(), learning_rate=lr, momentum=0.0)
        >>> config_ck = CheckpointConfig(save_checkpoint_steps=(int)(samples/cfg.batch_size),  keep_checkpoint_max=10)
        >>> model_instance = SuppressModel(network=networks_l5,
        >>>                            loss_fn=net_loss,
        >>>                            optimizer=net_opt,
        >>>                            metrics={"Accuracy": Accuracy()})
        >>> model_instance.link_suppress_ctrl(suppress_ctrl_instance)
        >>> ds_train = generate_mnist_dataset("./MNIST_unzip/train",
        >>>                                 batch_size=cfg.batch_size, repeat_size=1, samples=samples)
        >>> ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet",
        >>>                          directory="./trained_ckpt_file/",
        >>>                          config=config_ck)
        >>> model_instance.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(), suppress_masker],
        >>>                  dataset_sink_mode=False)
    """

    def __init__(self,
                 network,
                 loss_fn,
                 optimizer,
                 **kwargs):

        check_param_type('network', network, Cell)
        check_param_type('optimizer', optimizer, SGD)

        self.network_end = None
        self._train_one_step = None

        super(SuppressModel, self).__init__(network, loss_fn, optimizer, **kwargs)

    def link_suppress_ctrl(self, suppress_pri_ctrl):
        """
        Link self and SuppressCtrl instance.

        Args:
            suppress_pri_ctrl (SuppressCtrl): SuppressCtrl instance.
        """
        check_param_type('suppress_pri_ctrl', suppress_pri_ctrl, SuppressCtrl)

        suppress_pri_ctrl.model = self
        if self._train_one_step is not None:
            self._train_one_step.link_suppress_ctrl(suppress_pri_ctrl)

    def _build_train_network(self):
        """Build train network"""
        network = self._network

        ms_mode = context.get_context("mode")
        if ms_mode != context.PYNATIVE_MODE:
            raise ValueError("Only PYNATIVE_MODE is supported for suppress privacy now.")

        if self._optimizer:
            network = self._amp_build_train_network(network,
                                                    self._optimizer,
                                                    self._loss_fn,
                                                    level=self._amp_level,
                                                    keep_batchnorm_fp32=self._keep_bn_fp32)
        else:
            raise ValueError("_optimizer is none")

        self._train_one_step = network

        if self._parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL,
                                   ParallelMode.AUTO_PARALLEL):
            network.set_auto_parallel()

        self.network_end = self._train_one_step.network
        return network

    def _amp_build_train_network(self, network, optimizer, loss_fn=None,
                                 level='O0', **kwargs):
        """
        Build the mixed precision training cell automatically.

        Args:
            network (Cell): Definition of the network.
            loss_fn (Union[None, Cell]): Definition of the loss_fn. If None,
                the `network` should have the loss inside. Default: None.
            optimizer (Optimizer): Optimizer to update the Parameter.
            level (str): Supports [O0, O2]. Default: "O0".
                - O0: Do not change.
                - O2: Cast network to float16, keep batchnorm and `loss_fn`
                  (if set) run in float32, using dynamic loss scale.
            cast_model_type (:class:`mindspore.dtype`): Supports `mstype.float16`
                or `mstype.float32`. If set to `mstype.float16`, use `float16`
                mode to train. If set, overwrite the level setting.
            keep_batchnorm_fp32 (bool): Keep Batchnorm run in `float32`. If set,
                overwrite the level setting.
            loss_scale_manager (Union[None, LossScaleManager]): If None, not
                scale the loss, or else scale the loss by LossScaleManager.
                If set, overwrite the level setting.
        """
        validator.check_value_type('network', network, nn.Cell, None)
        validator.check_value_type('optimizer', optimizer, nn.Optimizer, None)
        validator.check('level', level, "", ['O0', 'O2'], Rel.IN, None)
        self._check_kwargs(kwargs)
        config = dict(_config_level[level], **kwargs)
        config = edict(config)

        if config.cast_model_type == mstype.float16:
            network.to_float(mstype.float16)

            if config.keep_batchnorm_fp32:
                _do_keep_batchnorm_fp32(network)

        if loss_fn:
            network = _add_loss_network(network, loss_fn,
                                        config.cast_model_type)

        if _get_parallel_mode() in (
                ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            network = _VirtualDatasetCell(network)

        loss_scale = 1.0
        if config.loss_scale_manager is not None:
            print("----model config have loss scale manager !")
        network = TrainOneStepCell(network, optimizer, sens=loss_scale).set_train()
        return network


class _TupleAdd(nn.Cell):
    """
    Add two tuple of data.
    """
    def __init__(self):
        super(_TupleAdd, self).__init__()
        self.add = P.Add()
        self.hyper_map = C.HyperMap()

    def construct(self, input1, input2):
        """Add two tuple of data."""
        out = self.hyper_map(self.add, input1, input2)
        return out

class _TupleMul(nn.Cell):
    """
    Mul two tuple of data.
    """
    def __init__(self):
        super(_TupleMul, self).__init__()
        self.mul = P.Mul()
        self.hyper_map = C.HyperMap()

    def construct(self, input1, input2):
        """Add two tuple of data."""
        out = self.hyper_map(self.mul, input1, input2)
        #print(out)
        return out

# come from nn.cell_wrapper.TrainOneStepCell
class TrainOneStepCell(Cell):
    r"""
    Network training package class.

    Wraps the network with an optimizer. The resulting Cell be trained with input data and label.
    Backward graph will be created in the construct function to do parameter updating. Different
    parallel modes are available to run the training.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The scaling number to be filled as the input of backpropagation. Default value is 1.0.

    Inputs:
        - **data** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a scalar Tensor with shape :math:`()`.
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self._tuple_add = _TupleAdd()
        self._tuple_mul = _TupleMul()
        parallel_mode = _get_parallel_mode()
        if parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)

        self.do_privacy = False
        self.grad_mask_tup = ()    # tuple containing grad_mask(cell)
        self.de_weight_tup = ()    # tuple containing de_weight(cell)
        self._suppress_pri_ctrl = None

    def link_suppress_ctrl(self, suppress_pri_ctrl):
        """
        Set Suppress Mask for grad_mask_tup and de_weight_tup.

        Args:
           suppress_pri_ctrl (SuppressCtrl): SuppressCtrl instance.
        """
        self._suppress_pri_ctrl = suppress_pri_ctrl
        if self._suppress_pri_ctrl.grads_mask_list:
            for grad_mask_cell in self._suppress_pri_ctrl.grads_mask_list:
                self.grad_mask_tup += (grad_mask_cell,)
                self.do_privacy = True
            for de_weight_cell in self._suppress_pri_ctrl.de_weight_mask_list:
                self.de_weight_tup += (de_weight_cell,)
        else:
            self.do_privacy = False

    def construct(self, data, label):
        """
        Construct a compute flow.
        """
        weights = self.weights
        loss = self.network(data, label)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(data, label, sens)

        new_grads = ()
        m = 0
        for grad in grads:
            if self.do_privacy and self._suppress_pri_ctrl.mask_started:
                enable_mask, grad_mask = self.grad_mask_tup[m]()
                enable_de_weight, de_weight_array = self.de_weight_tup[m]()

                if enable_mask and enable_de_weight:
                    grad_n = self._tuple_add(de_weight_array, self._tuple_mul(grad, grad_mask))
                    new_grads = new_grads + (grad_n,)
                else:
                    new_grads = new_grads + (grad,)
            else:
                new_grads = new_grads + (grad,)
            m = m + 1

        if self.reducer_flag:
            new_grads = self.grad_reducer(new_grads)

        return F.depend(loss, self.optimizer(new_grads))
