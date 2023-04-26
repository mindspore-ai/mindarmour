mindarmour.privacy.sup_privacy
==============================

本模块提供抑制隐私功能，以保护用户隐私。

.. py:class:: mindarmour.privacy.sup_privacy.SuppressMasker(model, suppress_ctrl)

    周期性检查抑制隐私功能状态和切换（启动/关闭）抑制操作。

    详情请查看： `应用抑制隐私机制保护用户隐私
    <https://mindspore.cn/mindarmour/docs/zh-CN/master/protect_user_privacy_with_suppress_privacy.html#%E5%BC%95%E5%85%A5%E6%8A%91%E5%88%B6%E9%9A%90%E7%A7%81%E8%AE%AD%E7%BB%83>`_。

    参数：
        - **model** (SuppressModel) - SuppressModel 实例。
        - **suppress_ctrl** (SuppressCtrl) - SuppressCtrl 实例。

    .. py:method:: step_end(run_context)

        更新用于抑制模型实例的掩码矩阵张量。

        参数：
            - **run_context** (RunContext) - 包含模型的一些信息。

.. py:class:: mindarmour.privacy.sup_privacy.SuppressModel(network, loss_fn, optimizer, **kwargs)

    抑制隐私训练器，重载自 `mindspore.train.Model` 。

    有关详细信息，请查看： `应用抑制隐私机制保护用户隐私 <https://mindspore.cn/mindarmour/docs/zh-CN/master/protect_user_privacy_with_suppress_privacy.html#%E5%BC%95%E5%85%A5%E6%8A%91%E5%88%B6%E9%9A%90%E7%A7%81%E8%AE%AD%E7%BB%83>`_。

    参数：
        - **network** (Cell) - 要训练的神经网络模型。
        - **loss_fn** (Cell) - 优化器的损失函数。
        - **optimizer** (Optimizer) - 优化器实例。
        - **kwargs** - 创建抑制模型时使用的关键字参数。

    .. py:method:: link_suppress_ctrl(suppress_pri_ctrl)

        SuppressCtrl实例关联到SuppressModel实例。

        参数：
            - **suppress_pri_ctrl** (SuppressCtrl) - SuppressCtrl实例。

.. py:class:: mindarmour.privacy.sup_privacy.SuppressPrivacyFactory

    SuppressCtrl机制的工厂类。

    详情请查看： `应用抑制隐私机制保护用户隐私 <https://mindspore.cn/mindarmour/docs/zh-CN/master/protect_user_privacy_with_suppress_privacy.html#%E5%BC%95%E5%85%A5%E6%8A%91%E5%88%B6%E9%9A%90%E7%A7%81%E8%AE%AD%E7%BB%83>`_。

    .. py:method:: create(networks, mask_layers, policy='local_train', end_epoch=10, batch_num=20, start_epoch=3, mask_times=1000, lr=0.05, sparse_end=0.90, sparse_start=0.0)

        参数：
            - **networks** (Cell) - 要训练的神经网络模型。此网络参数应与SuppressModel()的'network'参数相同。
            - **mask_layers** (list) - 需要抑制的训练网络层的描述。
            - **policy** (str) - 抑制隐私训练的训练策略。默认值： ``"local_train"``，表示本地训练。
            - **end_epoch** (int) - 最后一次抑制操作对应的epoch序号，0<start_epoch<=end_epoch<=100。默认值：``10``。此参数应与 `mindspore.train.model.train()` 的 `epoch` 参数相同。
            - **batch_num** (int) - 一个epoch中批次的数量，应等于num_samples/batch_size。默认值：``20``。
            - **start_epoch** (int) - 第一个抑制操作对应的epoch序号，0<start_epoch<=end_epoch<=100。默认值：``3``。
            - **mask_times** (int) - 抑制操作的数量。默认值：``1000``。
            - **lr** (Union[float, int]) - 学习率，在训练期间应保持不变。0<lr<=0.50. 默认值：0.05。此lr参数应与 `mindspore.nn.SGD()` 的 `learning_rate` 参数相同。
            - **sparse_end** (float) - 要到达的稀疏性，0.0<=sparse_start<sparse_end<1.0。默认值：``0.90``。
            - **sparse_start** (Union[float, int]) - 抑制操作启动时对应的稀疏性，0.0<=sparse_start<sparse_end<1.0。默认值：``0.0``。

        返回：
            - **SuppressCtrl** - 抑制隐私机制的类。

.. py:class:: mindarmour.privacy.sup_privacy.SuppressCtrl(networks, mask_layers, end_epoch, batch_num, start_epoch, mask_times, lr, sparse_end, sparse_start)

    完成抑制隐私操作，包括计算抑制比例，找到应该抑制的参数，并永久抑制这些参数。

    详情请查看： `应用抑制隐私机制保护用户隐私 <https://mindspore.cn/mindarmour/docs/zh-CN/master/protect_user_privacy_with_suppress_privacy.html#%E5%BC%95%E5%85%A5%E6%8A%91%E5%88%B6%E9%9A%90%E7%A7%81%E8%AE%AD%E7%BB%83>`_。

    参数：
        - **networks** (Cell) - 要训练的神经网络模型。
        - **mask_layers** (list) - 需要抑制的层的描述。
        - **end_epoch** (int) - 最后一次抑制操作对应的epoch序号。
        - **batch_num** (int) - 一个epoch中的batch数量。
        - **start_epoch** (int) - 第一个抑制操作对应的epoch序号。
        - **mask_times** (int) - 抑制操作的数量。
        - **lr** (Union[float, int]) - 学习率。
        - **sparse_end** (float) - 要到达的稀疏性。
        - **sparse_start** (Union[float, int]) - 要启动的稀疏性。

    .. py:method:: calc_actual_sparse_for_conv(networks)

        计算con1层和con2层的网络稀疏性。

        参数：
            - **networks** (Cell) - 要训练的神经网络模型。

    .. py:method:: calc_actual_sparse_for_fc1(networks)

        计算全连接1层的实际稀疏

        参数：
            - **networks** (Cell) - 要训练的神经网络模型。

    .. py:method:: calc_actual_sparse_for_layer(networks, layer_name)

        计算一个网络层的实际稀疏性

        参数：
            - **networks** (Cell) - 要训练的神经网络模型。
            - **layer_name** (str) - 目标层的名称。

    .. py:method:: calc_theoretical_sparse_for_conv()

        计算卷积层的掩码矩阵的实际稀疏性。

    .. py:method:: print_paras()

        显示参数信息

    .. py:method:: reset_zeros()

        将用于加法运算的掩码数组设置为0。

    .. py:method:: update_mask(networks, cur_step, target_sparse=0.0)

        对整个模型的用于加法运算和乘法运算的掩码数组进行更新。

        参数：
            - **networks** (Cell) - 训练网络。
            - **cur_step** (int) - 整个训练过程的当前epoch。
            - **target_sparse** (float) - 要到达的稀疏性。默认值：``0.0``。

    .. py:method:: update_mask_layer(weight_array_flat, sparse_weight_thd, sparse_stop_pos, weight_abs_max, layer_index)

        对单层的用于加法运算和乘法运算的掩码数组进行更新。

        参数：
            - **weight_array_flat** (numpy.ndarray) - 层参数权重数组。
            - **sparse_weight_thd** (float) - 绝对值小于该阈值的权重会被抑制。
            - **sparse_stop_pos** (int) - 要抑制的最大元素数。
            - **weight_abs_max** (float) - 权重的最大绝对值。
            - **layer_index** (int) - 目标层的索引。

    .. py:method:: update_mask_layer_approximity(weight_array_flat, weight_array_flat_abs, actual_stop_pos, layer_index)

        对单层的用于加法运算和乘法运算的掩码数组进行更新。

        禁用clipping lower、clipping、adding noise操作。

        参数：
            - **weight_array_flat** (numpy.ndarray) - 层参数权重数组。
            - **weight_array_flat_abs** (numpy.ndarray) - 层参数权重的绝对值的数组。
            - **actual_stop_pos** (int) - 应隐藏实际参数编号。
            - **layer_index** (int) - 目标层的索引。

    .. py:method:: update_status(cur_epoch, cur_step, cur_step_in_epoch)

        更新抑制操作状态。

        参数：
            - **cur_epoch** (int) - 整个训练过程的当前epoch。
            - **cur_step** (int) - 整个训练过程的当前步骤。
            - **cur_step_in_epoch** (int) - 当前epoch的当前步骤。

.. py:class:: mindarmour.privacy.sup_privacy.MaskLayerDes(layer_name, grad_idx, is_add_noise, is_lower_clip, min_num, upper_bound=1.20)

    对抑制目标层的描述。

    参数：
        - **layer_name** (str) - 层名称，如下获取一个层的名称：

          .. code-block::

              for layer in networks.get_parameters(expand=True):
                  if layer.name == "conv": ...

        - **grad_idx** (int) - 掩码层在梯度元组中的索引。可参考 `model.py <https://gitee.com/mindspore/mindarmour/blob/master/mindarmour/privacy/sup_privacy/train/model.py>`_ 中TrainOneStepCell的构造函数，在PYNATIVE_MODE模式下打印某些层的索引值。
        - **is_add_noise** (bool) - 如果为 ``True``，则此层的权重可以添加噪声。如果为 ``False``，则此层的权重不能添加噪声。如果参数num大于100000，则　`is_add_noise` 无效。
        - **is_lower_clip** (bool) - 如果为 ``True``，则此层的权重将被剪裁到大于下限值。如果为 ``False``，此层的权重不会被要求大于下限制。如果参数num大于100000，则 `is_lower_clip` 无效。
        - **min_num** (int) - 未抑制的剩余权重数。如果 `min_num` 小于（参数总数量 *　`SupperssCtrl.sparse_end` ），则 `min_num` 无效。
        - **upper_bound** (Union[float, int]) - 此层权重的最大abs值，默认值：``1.20``。如果参数num大于100000，则 `upper_bound` 无效。

