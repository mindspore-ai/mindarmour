mindarmour.fuzz_testing
=======================

该模块提供了一种基于神经元覆盖率增益的模糊测试方法来评估给定模型的鲁棒性。

.. py:class:: mindarmour.fuzz_testing.Fuzzer(target_model)

    深度神经网络的模糊测试框架。

    参考文献：`DeepHunter: A Coverage-Guided Fuzz Testing Framework for Deep Neural Networks <https://dl.acm.org/doi/10.1145/3293882.3330579>`_。

    参数：
        - **target_model** (Model) - 目标模糊模型。

    .. py:method:: fuzzing(mutate_config, initial_seeds, coverage, evaluate=True, max_iters=10000, mutate_num_per_seed=20)

        深度神经网络的模糊测试。

        参数：
            - **mutate_config** (list) - 变异方法配置。格式为：

              .. code-block:: python

                  mutate_config = [
                      {'method': 'GaussianBlur',
                       'params': {'ksize': [1, 2, 3, 5], 'auto_param': [True, False]}},
                      {'method': 'UniformNoise',
                       'params': {'factor': [0.1, 0.2, 0.3], 'auto_param': [False, True]}},
                      {'method': 'GaussianNoise',
                       'params': {'factor': [0.1, 0.2, 0.3], 'auto_param': [False, True]}},
                      {'method': 'Contrast',
                       'params': {'alpha': [0.5, 1, 1.5], 'beta': [-10, 0, 10], 'auto_param': [False, True]}},
                      {'method': 'Rotate',
                       'params': {'angle': [20, 90], 'auto_param': [False, True]}},
                      {'method': 'FGSM',
                       'params': {'eps': [0.3, 0.2, 0.4], 'alpha': [0.1], 'bounds': [(0, 1)]}}
                      ...]

              - 支持的方法在列表 `self._strategies` 中，每个方法的参数必须在可选参数的范围内。支持的方法分为两种类型：
              - 首先，自然鲁棒性方法包括：'Translate'、'Scale'、'Shear'、'Rotate'、'Perspective'、'Curve'、'GaussianBlur'、'MotionBlur'、'GradientBlur'、'Contrast'、'GradientLuminance'、'UniformNoise'、'GaussianNoise'、'SaltAndPepperNoise'、'NaturalNoise'。
              - 其次，对抗样本攻击方式包括：'FGSM'、'PGD'和'MDIM'。'FGSM'、'PGD'和'MDIM'分别是 FastGradientSignMethod、ProjectedGradientDent和MomentumDiverseInputIterativeMethod的缩写。 `mutate_config` 必须包含在['Contrast', 'GradientLuminance', 'GaussianBlur', 'MotionBlur', 'GradientBlur', 'UniformNoise', 'GaussianNoise', 'SaltAndPepperNoise', 'NaturalNoise']中的方法。

              - 第一类方法的参数设置方式可以在'mindarmour/natural_robustness/transform/image'中看到。第二类方法参数配置参考 `self._attack_param_checklists` 。
            - **initial_seeds** (list[list]) - 用于生成变异样本的初始种子队列。初始种子队列的格式为[[image_data, label], [...], ...]，且标签必须为one-hot。
            - **coverage** (CoverageMetrics) - 神经元覆盖率指标类。
            - **evaluate** (bool) - 是否返回评估报告。默认值：``True``。
            - **max_iters** (int) - 选择要变异的种子的最大数量。默认值：``10000``。
            - **mutate_num_per_seed** (int) - 每个种子的最大变异次数。默认值：``20``。

        返回：
            - **list** - 模糊测试生成的变异样本。
            - **list** - 变异样本的ground truth标签。
            - **list** - 预测结果。
            - **list** - 变异策略。
            - **dict** - Fuzzer的指标报告。

        异常：
            - **ValueError** - 参数 `coverage` 必须是CoverageMetrics的子类。
            - **ValueError** - 初始种子队列为空。
            - **ValueError** - `initial_seeds` 中的种子未包含两个元素。

.. py:class:: mindarmour.fuzz_testing.CoverageMetrics(model, incremental=False, batch_size=32)

    计算覆盖指标的神经元覆盖类的抽象基类。

    训练后网络的每个神经元输出有一个输出范围（我们称之为原始范围），测试数据集用于估计训练网络的准确性。然而，不同的测试数据集，神经元的输出分布会有所不同。因此，与传统模糊测试类似，模型模糊测试意味着测试这些神经元的输出，并评估在测试数据集上神经元输出值占原始范围的比例。

    参考文献： `DeepGauge: Multi-Granularity Testing Criteria for Deep Learning Systems <https://arxiv.org/abs/1803.07519>`_。

    参数：
        - **model** (Model) - 被测模型。
        - **incremental** (bool) - 指标将以增量方式计算。默认值：``False``。
        - **batch_size** (int) - 模糊测试批次中的样本数。默认值：``32``。
    
    .. py:method:: get_metrics(dataset)

        计算给定数据集的覆盖率指标。

        参数：
            - **dataset** (numpy.ndarray) - 用于计算覆盖指标的数据集。

        异常：
            - **NotImplementedError** - 抽象方法。

.. py:class:: mindarmour.fuzz_testing.NeuronCoverage(model, threshold=0.1, incremental=False, batch_size=32)

    计算神经元激活的覆盖率。当神经元的输出大于阈值时，神经元被激活。

    神经元覆盖率等于网络中激活的神经元占总神经元的比例。

    参数：
        - **model** (Model) - 被测模型。
        - **threshold** (float) - 用于确定神经元是否激活的阈值。默认值：``0.1``。
        - **incremental** (bool) - 指标将以增量方式计算。默认值：``False``。
        - **batch_size** (int) - 模糊测试批次中的样本数。默认值：``32``。

    .. py:method:: get_metrics(dataset)

        获取神经元覆盖率的指标：激活的神经元占网络中神经元总数的比例。

        参数：
            - **dataset** (numpy.ndarray) - 用于计算覆盖率指标的数据集。

        返回：
            - **float** - 'neuron coverage'的指标。

.. py:class:: mindarmour.fuzz_testing.TopKNeuronCoverage(model, top_k=3, incremental=False, batch_size=32)

    计算前k个激活神经元的覆盖率。当隐藏层神经元的输出值在最大的 `top_k` 范围内，神经元就会被激活。`top_k` 神经元覆盖率等于网络中激活神经元占总神经元的比例。

    参数：
        - **model** (Model) - 被测模型。
        - **top_k** (int) - 当隐藏层神经元的输出值在最大的 `top_k` 范围内，神经元就会被激活。默认值：``3``。
        - **incremental** (bool) - 指标将以增量方式计算。默认值：``False``。
        - **batch_size** (int) - 模糊测试批次中的样本数。默认值：``32``。

    .. py:method:: get_metrics(dataset)

        获取Top K激活神经元覆盖率的指标。

        参数：
            - **dataset** (numpy.ndarray) - 用于计算覆盖率指标的数据集。

        返回：
            - **float** - 'top k neuron coverage'的指标。

.. py:class:: mindarmour.fuzz_testing.NeuronBoundsCoverage(model, train_dataset, incremental=False, batch_size=32)

    获取'neuron boundary coverage'的指标 :math:`NBC = (|UpperCornerNeuron| + |LowerCornerNeuron|)/(2*|N|)` ，其中 :math:`|N|` 是神经元的数量，NBC是指测试数据集中神经元输出值超过训练数据集中相应神经元输出值的上下界的神经元比例。

    参数：
        - **model** (Model) - 等待测试的预训练模型。
        - **train_dataset** (numpy.ndarray) - 用于确定神经元输出边界的训练数据集。
        - **incremental** (bool) - 指标将以增量方式计算。默认值：``False``。
        - **batch_size** (int) - 模糊测试批次中的样本数。默认值：``32``。

    .. py:method:: get_metrics(dataset)

        获取'neuron boundary coverage'的指标。

        参数：
            - **dataset** (numpy.ndarray) - 用于计算覆盖指标的数据集。

        返回：
            - **float** - 'neuron boundary coverage'的指标。

.. py:class:: mindarmour.fuzz_testing.SuperNeuronActivateCoverage(model, train_dataset, incremental=False, batch_size=32)

    获取超激活神经元覆盖率（'super neuron activation coverage'）的指标。 :math:`SNAC = |UpperCornerNeuron|/|N|` 。SNAC是指测试集中神经元输出值超过训练集中相应神经元输出值上限的神经元比例。

    参数：
        - **model** (Model) - 等待测试的预训练模型。
        - **train_dataset** (numpy.ndarray) - 用于确定神经元输出边界的训练数据集。
        - **incremental** (bool) - 指标将以增量方式计算。默认值：``False``。
        - **batch_size** (int) - 模糊测试批次中的样本数。默认值：``32``。

    .. py:method:: get_metrics(dataset)

        获取超激活神经元覆盖率（'super neuron activation coverage'）的指标。

        参数：
            - **dataset** (numpy.ndarray) - 用于计算覆盖指标的数据集。

        返回：
            - **float** - 超激活神经元覆盖率（'super neuron activation coverage'）的指标

.. py:class:: mindarmour.fuzz_testing.KMultisectionNeuronCoverage(model, train_dataset, segmented_num=100, incremental=False, batch_size=32)

    获取K分神经元覆盖率的指标。KMNC度量测试集神经元输出落在训练集输出范围k等分间隔上的比例。

    参数：
        - **model** (Model) - 等待测试的预训练模型。
        - **train_dataset** (numpy.ndarray) - 用于确定神经元输出边界的训练数据集。
        - **segmented_num** (int) - 神经元输出间隔的分段部分数量。默认值：``100``。
        - **incremental** (bool) - 指标将以增量方式计算。默认值：``False``。
        - **batch_size** (int) - 模糊测试批次中的样本数。默认值：``32``。

    .. py:method:: get_metrics(dataset)

        获取'k-multisection neuron coverage'的指标。

        参数：
            - **dataset** (numpy.ndarray) - 用于计算覆盖指标的数据集。

        返回：
            - **float** - 'k-multisection neuron coverage'的指标。
