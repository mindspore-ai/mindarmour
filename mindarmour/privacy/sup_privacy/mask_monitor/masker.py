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
Masker module of suppress-based privacy..
"""
from mindspore.train.callback import Callback
from mindarmour.utils.logger import LogUtil
from mindarmour.utils._check_param import check_param_type
from mindarmour.privacy.sup_privacy.train.model import SuppressModel
from mindarmour.privacy.sup_privacy.sup_ctrl.conctrl import SuppressCtrl

LOGGER = LogUtil.get_instance()
TAG = 'suppress masker'


class SuppressMasker(Callback):
    """
    Periodicity check suppress privacy function status and toggle suppress operation.
    For details, please check `Protecting User Privacy with Suppression Privacy
    <https://mindspore.cn/mindarmour/docs/en/r1.8/protect_user_privacy_with_suppress_privacy.html>`_.

    Args:
        model (SuppressModel):  SuppressModel instance.
        suppress_ctrl (SuppressCtrl): SuppressCtrl instance.

    Examples:
        >>> import mindspore.nn as nn
        >>> import mindspore.ops.operations as P
        >>> from mindspore import context
        >>> from mindspore.nn import Accuracy
        >>> from mindarmour.privacy.sup_privacy import SuppressModel
        >>> from mindarmour.privacy.sup_privacy import SuppressMasker
        >>> from mindarmour.privacy.sup_privacy import SuppressPrivacyFactory
        >>> from mindarmour.privacy.sup_privacy import MaskLayerDes
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self._softmax = P.Softmax()
        ...         self._Dense = nn.Dense(10,10)
        ...         self._squeeze = P.Squeeze(1)
        ...     def construct(self, inputs):
        ...         out = self._softmax(inputs)
        ...         out = self._Dense(out)
        ...         return self._squeeze(out)
        >>> context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
        >>> network = Net()
        >>> masklayers = []
        >>> masklayers.append(MaskLayerDes("_Dense.weight", 0, False, True, 10))
        >>> suppress_ctrl_instance = SuppressPrivacyFactory().create(networks=network,
        ...                                                          mask_layers=masklayers,
        ...                                                          policy="local_train",
        ...                                                          end_epoch=10,
        ...                                                          batch_num=1,
        ...                                                          start_epoch=3,
        ...                                                          mask_times=10,
        ...                                                          lr=0.05,
        ...                                                          sparse_end=0.95,
        ...                                                          sparse_start=0.0)
        >>> net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        >>> net_opt = nn.SGD(network.trainable_params(), 0.05)
        >>> model_instance = SuppressModel(network=network,
        ...                                loss_fn=net_loss,
        ...                                optimizer=net_opt,
        ...                                metrics={"Accuracy": Accuracy()})
        >>> model_instance.link_suppress_ctrl(suppress_ctrl_instance)
        >>> masker_instance = SuppressMasker(model_instance, suppress_ctrl_instance)
    """

    def __init__(self, model, suppress_ctrl):

        super(SuppressMasker, self).__init__()

        self._model = check_param_type('model', model, SuppressModel)
        self._suppress_ctrl = check_param_type('suppress_ctrl', suppress_ctrl, SuppressCtrl)

    def step_end(self, run_context):
        """
        Update mask matrix tensor used for SuppressModel instance.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if self._suppress_ctrl is not None and self._model.network_end is not None:
            if not self._suppress_ctrl.mask_initialized:
                raise ValueError("Not initialize network!")
            if cur_step_in_epoch % 100 == 1:
                self._suppress_ctrl.calc_theoretical_sparse_for_conv()
                _, _, _ = self._suppress_ctrl.calc_actual_sparse_for_conv(
                    self._suppress_ctrl.networks)
            self._suppress_ctrl.update_status(cb_params.cur_epoch_num, cur_step, cur_step_in_epoch)
            if self._suppress_ctrl.to_do_mask:
                self._suppress_ctrl.update_mask(self._suppress_ctrl.networks, cur_step)
                LOGGER.info(TAG, "suppress update")
            elif not self._suppress_ctrl.to_do_mask and self._suppress_ctrl.mask_started:
                self._suppress_ctrl.reset_zeros()
