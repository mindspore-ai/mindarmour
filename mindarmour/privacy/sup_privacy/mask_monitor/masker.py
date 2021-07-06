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
    Args:
        model (SuppressModel):  SuppressModel instance.
        suppress_ctrl (SuppressCtrl): SuppressCtrl instance.

    Examples:
        >>> networks_l5 = LeNet5()
        >>> masklayers = []
        >>> masklayers.append(MaskLayerDes("conv1.weight", 0, False, True, 10))
        >>> suppress_ctrl_instance = SuppressPrivacyFactory().create(networks=networks_l5,
        >>>                                                     mask_layers=masklayers,
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
