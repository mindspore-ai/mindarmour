mindarmour.privacy.diff_privacy
===============================

本模块提供差分隐私功能，以保护用户隐私。

.. py:class:: mindarmour.privacy.diff_privacy.NoiseGaussianRandom(norm_bound=1.0, initial_noise_multiplier=1.0, seed=0, decay_policy=None)

    基于 :math:`mean=0` 以及 :math:`standard\_deviation = norm\_bound * initial\_noise\_multiplier` 的高斯分布产生噪声。

    参数：
        - **norm_bound** (float) - 梯度的l2范数的裁剪范围。默认值：1.0。
        - **initial_noise_multiplier** (float) - 高斯噪声标准偏差除以 `norm_bound` 的比率，将用于计算隐私预算。默认值：1.0。
        - **seed** (int) - 原始随机种子，如果seed=0随机正态将使用安全随机数。如果seed!=0随机正态将使用给定的种子生成值。默认值：0。
        - **decay_policy** (str) - 衰减策略。默认值：None。

    .. py:method:: construct(gradients)

        产生高斯噪声。

        参数：
            - **gradients** (Tensor) - 梯度。

        返回：
            - **Tensor** - 生成的shape与给定梯度相同的噪声。

.. py:class:: mindarmour.privacy.diff_privacy.NoiseAdaGaussianRandom(norm_bound=1.0, initial_noise_multiplier=1.0, seed=0, noise_decay_rate=6e-6, decay_policy='Exp')

    自适应高斯噪声产生机制。噪音会随着训练而衰减。衰减模式可以是'Time'、'Step'、'Exp'。
    在模型训练过程中，将更新 `self._noise_multiplier` 。

    参数：
        - **norm_bound** (float) - 梯度的l2范数的裁剪范围。默认值：1.0。
        - **initial_noise_multiplier** (float) - 高斯噪声标准偏差除以 `norm_bound` 的比率，将用于计算隐私预算。默认值：1.0。
        - **seed** (int) - 原始随机种子，如果seed=0随机正态将使用安全随机数。如果seed!=0随机正态将使用给定的种子生成值。默认值：0。
        - **noise_decay_rate** (float) - 控制噪声衰减的超参数。默认值：6e-6。
        - **decay_policy** (str) - 噪声衰减策略包括'Step'、'Time'、'Exp'。默认值：'Exp'。

    .. py:method:: construct(gradients)

        生成自适应高斯噪声。

        参数：
            - **gradients** (Tensor) - 梯度。

        返回：
            - **Tensor** - 生成的shape与给定梯度相同的噪声。

.. py:class:: mindarmour.privacy.diff_privacy.AdaClippingWithGaussianRandom(decay_policy='Linear', learning_rate=0.001, target_unclipped_quantile=0.9, fraction_stddev=0.01, seed=0)

    自适应剪裁。
    如果 `decay_policy` 是'Linear'，则更新公式为：:math:`norm\_bound = norm\_bound - learning\_rate*(beta - target\_unclipped\_quantile)` 。

    如果 `decay_policy` 是'Geometric'，则更新公式为 :math:`norm\_bound = norm\_bound*exp(-learning\_rate*(empirical\_fraction - target\_unclipped\_quantile))` 。

    其中，beta是值最多为 `target_unclipped_quantile` 的样本的经验分数。

    参数：
        - **decay_policy** (str) - 自适应剪裁的衰变策略， `decay_policy` 必须在['Linear', 'Geometric']中。默认值：'Linear'。
        - **learning_rate** (float) - 更新范数裁剪的学习率。默认值：0.001。
        - **target_unclipped_quantile** (float) - 范数裁剪的目标分位数。默认值：0.9。
        - **fraction_stddev** (float) - 高斯正态的stddev，用于 `empirical_fraction` ，公式为empirical_fraction + N(0, fraction_stddev)。默认值：0.01。
        - **seed** (int) - 原始随机种子，如果seed=0随机正态将使用安全随机数。如果seed!=0随机正态将使用给定的种子生成值。默认值：0。

    返回：
        - **Tensor** - 更新后的梯度裁剪阈值。

    .. py:method:: construct(empirical_fraction, norm_bound)

        更新 `norm_bound` 的值。

        参数：
            - **empirical_fraction** (Tensor) - 梯度裁剪的经验分位数,最大值不超过 `target_unclipped_quantile` 。
            - **norm_bound** (Tensor) - 梯度的l2范数的裁剪范围。

        返回：
            - **Tensor** - 生成的shape与给定梯度相同的噪声。

.. py:class:: mindarmour.privacy.diff_privacy.NoiseMechanismsFactory

    噪声产生机制的工厂类。它目前支持高斯随机噪声（Gaussian Random Noise）和自适应高斯随机噪声（Adaptive Gaussian Random Noise）。

    详情请查看： `教程 <https://mindspore.cn/mindarmour/docs/zh-CN/r1.9/protect_user_privacy_with_differential_privacy.html#引入差分隐私>`_。

    .. py:method:: create(mech_name, norm_bound=1.0, initial_noise_multiplier=1.0, seed=0, noise_decay_rate=6e-6, decay_policy=None)

        参数：
            - **mech_name** (str) - 噪声生成策略，可以是'Gaussian'或'AdaGaussian'。噪声在'AdaGaussian'机制下衰减，而在'Gaussian'机制下则恒定。
            - **norm_bound** (float) - 梯度的l2范数的裁剪范围。默认值：1.0。
            - **initial_noise_multiplier** (float) - 高斯噪声标准偏差除以 `norm_bound` 的比率，将用于计算隐私预算。默认值：1.0。
            - **seed** (int) - 原始随机种子，如果seed=0随机正态将使用安全随机数。如果seed!=0随机正态将使用给定的种子生成值。默认值：0。
            - **noise_decay_rate** (float) - 控制噪声衰减的超参数。默认值：6e-6。
            - **decay_policy** (str) - 衰减策略。如果decay_policy为None，则不需要更新参数。默认值：None。

        异常：
            - **NameError** - `mech_name` 必须在['Gaussian', 'AdaGaussian']中。

        返回：
            - **Mechanisms** - 产生的噪声类别机制。

.. py:class:: mindarmour.privacy.diff_privacy.ClipMechanismsFactory

    梯度剪裁机制的工厂类。它目前支持高斯随机噪声（Gaussian Random Noise）的自适应剪裁（Adaptive Clipping）。

    详情请查看： `教程 <https://mindspore.cn/mindarmour/docs/zh-CN/r1.9/protect_user_privacy_with_differential_privacy.html#引入差分隐私>`_。

    .. py:method:: create(mech_name, decay_policy='Linear', learning_rate=0.001, target_unclipped_quantile=0.9, fraction_stddev=0.01, seed=0)

        参数：
            - **mech_name** (str) - 噪声裁剪生成策略，现支持'Gaussian'。
            - **decay_policy** (str) - 自适应剪裁的衰变策略，decay_policy必须在['Linear', 'Geometric']中。默认值：Linear。
            - **learning_rate** (float) - 更新范数裁剪的学习率。默认值：0.001。
            - **target_unclipped_quantile** (float) - 范数裁剪的目标分位数。默认值：0.9。
            - **fraction_stddev** (float) - 高斯正态的stddev，用于empirical_fraction，公式为 :math:`empirical\_fraction + N(0, fraction\_stddev)` 。默认值：0.01。
            - **seed** (int) - 原始随机种子，如果seed=0随机正态将使用安全随机数。如果seed!=0随机正态将使用给定的种子生成值。默认值：0。

        异常：
            - **NameError** - `mech_name` 必须在['Gaussian']中。

        返回：
            - **Mechanisms** - 产生的噪声类别机制。

.. py:class:: mindarmour.privacy.diff_privacy.PrivacyMonitorFactory

    DP训练隐私监视器的工厂类。

    详情请查看： `教程 <https://mindspore.cn/mindarmour/docs/zh-CN/r1.9/protect_user_privacy_with_differential_privacy.html#引入差分隐私>`_。

    .. py:method:: create(policy, *args, **kwargs)

        创建隐私预算监测类。

        参数：
            - **policy** (str) - 监控策略，现支持'rdp'和'zcdp'。

              - 如果策略为'rdp'，监控器将根据Renyi差分隐私（Renyi differential privacy，RDP）理论计算DP训练的隐私预算；
              - 如果策略为'zcdp'，监控器将根据零集中差分隐私（zero-concentrated differential privacy，zCDP）理论计算DP训练的隐私预算。注意，'zcdp'不适合子采样噪声机制。
            - **args** (Union[int, float, numpy.ndarray, list, str]) - 用于创建隐私监视器的参数。
            - **kwargs** (Union[int, float, numpy.ndarray, list, str]) - 用于创建隐私监视器的关键字参数。

        返回：
            - **Callback** - 隐私监视器。

.. py:class:: mindarmour.privacy.diff_privacy.RDPMonitor(num_samples, batch_size, initial_noise_multiplier=1.5, max_eps=10.0, target_delta=1e-3, max_delta=None, target_eps=None, orders=None, noise_decay_mode='Time', noise_decay_rate=6e-4, per_print_times=50, dataset_sink_mode=False)

    基于Renyi差分隐私（RDP）理论，计算DP训练的隐私预算。根据下面的参考文献，如果随机化机制被认为具有α阶的ε'-Renyi差分隐私，它也满足常规差分隐私(ε, δ)，如下所示：

    .. math::
        (ε'+\frac{log(1/δ)}{α-1}, δ)

    详情请查看： `教程 <https://mindspore.cn/mindarmour/docs/zh-CN/r1.9/protect_user_privacy_with_differential_privacy.html#引入差分隐私>`_。

    参考文献： `Rényi Differential Privacy of the Sampled Gaussian Mechanism <https://arxiv.org/abs/1908.10530>`_。

    参数：
        - **num_samples** (int) - 训练数据集中的样本总数。
        - **batch_size** (int) - 训练时批处理中的样本数。
        - **initial_noise_multiplier** (Union[float, int]) - 高斯噪声标准偏差除以norm_bound的比率，将用于计算隐私预算。默认值：1.5。
        - **max_eps** (Union[float, int, None]) - DP训练的最大可接受epsilon预算，用于估计最大训练epoch。'None'表示epsilon预算没有限制。默认值：10.0。
        - **target_delta** (Union[float, int, None]) - DP训练的目标delta预算。如果 `target_delta` 设置为δ，则隐私预算δ将在整个训练过程中是固定的。默认值：1e-3。
        - **max_delta** (Union[float, int, None]) - DP训练的最大可接受delta预算，用于估计最大训练epoch。 `max_delta` 必须小于1，建议小于1e-3，否则会溢出。'None'表示delta预算没有限制。默认值：None。
        - **target_eps** (Union[float, int, None]) - DP训练的目标epsilon预算。如果target_eps设置为ε，则隐私预算ε将在整个训练过程中是固定的。默认值：None。
        - **orders** (Union[None, list[int, float]]) - 用于计算rdp的有限阶数，必须大于1。不同阶的隐私预算计算结果会有所不同。为了获得更严格（更小）的隐私预算估计，可以尝试阶列表。默认值：None。
        - **noise_decay_mode** (Union[None, str]) - 训练时添加噪音的衰减模式，可以是None、'Time'、'Step'、'Exp'。默认值：'Time'。
        - **noise_decay_rate** (float) - 训练时噪音的衰变率。默认值：6e-4。
        - **per_print_times** (int) - 计算和打印隐私预算的间隔步数。默认值：50。
        - **dataset_sink_mode** (bool) - 如果为True，所有训练数据都将一次性传递到设备（Ascend）。如果为False，则训练数据将在每步训练后传递到设备。默认值：False。

    .. py:method:: max_epoch_suggest()

        估计最大训练epoch，以满足预定义的隐私预算。

        返回：
            - **int** - 建议的最大训练epoch。

    .. py:method:: step_end(run_context)

        在每个训练步骤后计算隐私预算。

        参数：
            - **run_context** (RunContext) - 包含模型的一些信息。

.. py:class:: mindarmour.privacy.diff_privacy.ZCDPMonitor(num_samples, batch_size, initial_noise_multiplier=1.5, max_eps=10.0, target_delta=1e-3, noise_decay_mode='Time', noise_decay_rate=6e-4, per_print_times=50, dataset_sink_mode=False)

    基于零集中差分隐私（zCDP）理论，计算DP训练的隐私预算。根据下面的参考文献，如果随机化机制满足ρ-zCDP机制，它也满足传统的差分隐私（ε, δ），如下所示：

    .. math::
        (ρ+２\sqrt{ρ*log(1/δ)}, δ)

    注意，ZCDPMonitor不适合子采样噪声机制（如NoiseAdaGaussianRandom和NoiseGaussianRandom）。未来将开发zCDP的匹配噪声机制。

    详情请查看：`教程 <https://mindspore.cn/mindarmour/docs/zh-CN/r1.9/protect_user_privacy_with_differential_privacy.html#引入差分隐私>`_。

    参考文献：`Concentrated Differentially Private Gradient Descent with Adaptive per-Iteration Privacy Budget <https://arxiv.org/abs/1808.09501>`_。

    参数：
        - **num_samples** (int) - 训练数据集中的样本总数。
        - **batch_size** (int) - 训练时批处理中的样本数。
        - **initial_noise_multiplier** (Union[float, int]) - 高斯噪声标准偏差除以norm_bound的比率，将用于计算隐私预算。默认值：1.5。
        - **max_eps** (Union[float, int]) - DP训练的最大可接受epsilon预算，用于估计最大训练epoch。默认值：10.0。
        - **target_delta** (Union[float, int]) - DP训练的目标delta预算。如果 `target_delta` 设置为δ，则隐私预算δ将在整个训练过程中是固定的。默认值：1e-3。
        - **noise_decay_mode** (Union[None, str]) - 训练时添加噪音的衰减模式，可以是None、'Time'、'Step'、'Exp'。默认值：'Time'。
        - **noise_decay_rate** (float) - 训练时噪音的衰变率。默认值：6e-4。
        - **per_print_times** (int) - 计算和打印隐私预算的间隔步数。默认值：50。
        - **dataset_sink_mode** (bool) - 如果为True，所有训练数据都将一次性传递到设备（Ascend）。如果为False，则训练数据将在每步训练后传递到设备。默认值：False。

    .. py:method:: max_epoch_suggest()

        估计最大训练epoch，以满足预定义的隐私预算。

        返回：
            - **int** - 建议的最大训练epoch。

    .. py:method:: step_end(run_context)

        在每个训练步骤后计算隐私预算。

        参数：
            - **run_context** (RunContext) - 包含模型的一些信息。

.. py:class:: mindarmour.privacy.diff_privacy.DPOptimizerClassFactory(micro_batches=2)

    优化器的工厂类。

    参数：
        - **micro_batches** (int) - 从原始批次拆分的小批次中的样本数量。默认值：2。

    返回：
        - **Optimizer** - 优化器类。

    .. py:method:: create(policy)

        创建DP优化器。策略可以是'sgd'、'momentum'、'adam'。

        参数：
            - **policy** (str) - 选择原始优化器类型。

        返回：
            - **Optimizer** - 一个带有差分加噪的优化器。

    .. py:method:: set_mechanisms(policy, *args, **kwargs)

        获取噪音机制对象。策略可以是'Gaussian'或'AdaGaussian'。候选的args和kwargs可以在mechanisms.py
        的 :class:`NoiseMechanismsFactory` 类中看到。

        参数：
            - **policy** (str) - 选择机制类型。

.. py:class:: mindarmour.privacy.diff_privacy.DPModel(micro_batches=2, norm_bound=1.0, noise_mech=None, clip_mech=None, optimizer=nn.Momentum, **kwargs)

    DPModel用于构建差分隐私训练的模型。
    
    这个类重载自 :class:`mindspore.Model` 。

    详情请查看： `教程 <https://mindspore.cn/mindarmour/docs/zh-CN/r1.9/protect_user_privacy_with_differential_privacy.html#引入差分隐私>`_。

    参数：
        - **micro_batches** (int) - 从原始批次拆分的小批次数。默认值：2。
        - **norm_bound** (float) - 用于裁剪范围，如果设置为1，将返回原始数据。默认值：1.0。
        - **noise_mech** (Mechanisms) - 用于生成不同类型的噪音。默认值：None。
        - **clip_mech** (Mechanisms) - 用于更新自适应剪裁。默认值：None。
        - **optimizer** (Cell) - 用于更新差分隐私训练过程中的模型权重值。默认值：nn.Momentum。

    异常：
        - **ValueError** - optimizer值为None。
        - **ValueError** - optimizer不是DPOptimizer，且noise_mech为None。
        - **ValueError** - optimizer是DPOptimizer，且noise_mech非None。
        - **ValueError** - noise_mech或DPOptimizer的mech方法是自适应的，而clip_mech不是None。
