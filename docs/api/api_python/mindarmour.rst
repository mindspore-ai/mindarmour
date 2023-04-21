mindarmour
==========

MindArmour是MindSpore的工具箱，用于增强模型可信，实现隐私保护机器学习。

.. py:class:: mindarmour.Attack

    所有通过创建对抗样本的攻击类的抽象基类。

    对抗样本是通过向原始样本添加对抗噪声来生成的。

    .. py:method:: batch_generate(inputs, labels, batch_size=64)

        根据输入样本及其标签来批量生成对抗样本。

        参数：
            - **inputs** (Union[numpy.ndarray, tuple]) - 生成对抗样本的原始样本。
            - **labels** (Union[numpy.ndarray, tuple]) - 原始/目标标签。若每个输入有多个标签，将它包装在元组中。
            - **batch_size** (int) - 一个批次中的样本数。默认值：64。

        返回：
            - **numpy.ndarray** - 生成的对抗样本。

    .. py:method:: generate(inputs, labels)

        根据正常样本及其标签生成对抗样本。

        参数：
            - **inputs** (Union[numpy.ndarray, tuple]) - 生成对抗样本的原始样本。
            - **labels** (Union[numpy.ndarray, tuple]) - 原始/目标标签。若每个输入有多个标签，将它包装在元组中。

        异常：
            - **NotImplementedError** - 此为抽象方法。

.. py:class:: mindarmour.BlackModel

    将目标模型视为黑盒的抽象类。模型应由用户定义。

    .. py:method:: is_adversarial(data, label, is_targeted)

        检查输入样本是否为对抗样本。

        参数：
            - **data** (numpy.ndarray) - 要检查的输入样本，通常是一些恶意干扰的样本。
            - **label** (numpy.ndarray) - 对于目标攻击，标签是受扰动样本的预期标签。对于无目标攻击，标签是相应未扰动样本的原始标签。
            - **is_targeted** (bool) - 对于有目标/无目标攻击，请选择True/False。

        返回：
            - **bool** - 如果为True，则输入样本是对抗性的。如果为False，则输入样本不是对抗性的。

    .. py:method:: predict(inputs)

        使用用户指定的模型进行预测。预测结果的shape应该是 :math:`(m, n)`，其中n表示此模型分类的类数。

        参数：
            - **inputs** (numpy.ndarray) - 要预测的输入样本。

        异常：
            - **NotImplementedError** - 抽象方法未实现。

.. py:class:: mindarmour.Detector

    所有对抗样本检测器的抽象基类。

    .. py:method:: detect(inputs)

        从输入样本中检测对抗样本。

        参数：
            - **inputs** (Union[numpy.ndarray, list, tuple]) - 要检测的输入样本。

        异常：
            - **NotImplementedError** - 抽象方法未实现。

    .. py:method:: detect_diff(inputs)

        计算输入样本和去噪样本之间的差值。

        参数：
            - **inputs** (Union[numpy.ndarray, list, tuple]) - 要检测的输入样本。

        异常：
            - **NotImplementedError** - 抽象方法未实现。

    .. py:method:: fit(inputs, labels=None)

        拟合阈值，拒绝与去噪样本差异大于阈值的对抗样本。当应用于正常样本时，阈值由假正率决定。

        参数：
            - **inputs** (numpy.ndarray) - 用于计算阈值的输入样本。
            - **labels** (numpy.ndarray) - 训练数据的标签。默认值：None。

        异常：
            - **NotImplementedError** - 抽象方法未实现。

    .. py:method:: transform(inputs)

        过滤输入样本中的对抗性噪声。

        参数：
            - **inputs** (Union[numpy.ndarray, list, tuple]) - 要转换的输入样本。

        异常：
            - **NotImplementedError** - 抽象方法未实现。

.. py:class:: mindarmour.Defense(network)

    所有防御类的抽象基类，用于防御对抗样本。

    参数：
        - **network** (Cell) - 要防御的MindSpore风格的深度学习模型。

    .. py:method:: batch_defense(inputs, labels, batch_size=32, epochs=5)

        对输入进行批量防御操作。

        参数：
            - **inputs** (numpy.ndarray) - 生成对抗样本的原始样本。
            - **labels** (numpy.ndarray) - 输入样本的标签。
            - **batch_size** (int) - 一个批次中的样本数。默认值：32。
            - **epochs** (int) - epochs的数量。默认值：5。

        返回：
            - **numpy.ndarray** - `batch_defense` 操作的损失。

        异常：
            - **ValueError** - `batch_size` 为0。

    .. py:method:: defense(inputs, labels)

        对输入进行防御操作。

        参数：
            - **inputs** (numpy.ndarray) - 生成对抗样本的原始样本。
            - **labels** (numpy.ndarray) - 输入样本的标签。

        异常：
            - **NotImplementedError** - 抽象方法未实现。

.. py:class:: mindarmour.Fuzzer(target_model)

    深度神经网络的模糊测试框架。

    参考文献： `DeepHunter: A Coverage-Guided Fuzz Testing Framework for Deep Neural Networks <https://dl.acm.org/doi/10.1145/3293882.3330579>`_。

    参数：
        - **target_model** (Model) - 目标模糊模型。

    .. py:method:: fuzzing(mutate_config, initial_seeds, coverage, evaluate=True, max_iters=10000, mutate_num_per_seed=20)

        深度神经网络的模糊测试。

        参数：
            - **mutate_config** (list) - 变异方法配置。格式为：

              .. code-block:: python
   
                  mutate_config = 
                      [{'method': 'GaussianBlur',
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
                        'params': {'eps': [0.3, 0.2, 0.4], 'alpha': [0.1], 'bounds': [(0, 1)]}}]
                      ...]

              - 支持的方法在列表 `self._strategies` 中，每个方法的参数必须在可选参数的范围内。支持的方法分为两种类型：
              - 首先，自然鲁棒性方法包括：'Translate'、'Scale'、'Shear'、'Rotate'、'Perspective'、'Curve'、'GaussianBlur'、'MotionBlur'、'GradientBlur'、'Contrast'、'GradientLuminance'、'UniformNoise'、'GaussianNoise'、'SaltAndPepperNoise'、'NaturalNoise'。
              - 其次，对抗样本攻击方式包括：'FGSM'、'PGD'和'MDIM'。'FGSM'、'PGD'和'MDIM'分别是 FastGradientSignMethod、ProjectedGradientDent和MomentumDiverseInputIterativeMethod的缩写。 `mutate_config` 必须包含在['Contrast', 'GradientLuminance', 'GaussianBlur', 'MotionBlur', 'GradientBlur', 'UniformNoise', 'GaussianNoise', 'SaltAndPepperNoise', 'NaturalNoise']中的方法。

              - 第一类方法的参数设置方式可以在 `mindarmour/natural_robustness/transform/image <https://gitee.com/mindspore/mindarmour/tree/r2.0/mindarmour/natural_robustness/transform/image>`_ 中看到。第二类方法参数配置参考 `self._attack_param_checklists` 。
            - **initial_seeds** (list[list]) - 用于生成变异样本的初始种子队列。初始种子队列的格式为[[image_data, label], [...], ...]，且标签必须为one-hot。
            - **coverage** (CoverageMetrics) - 神经元覆盖率指标类。
            - **evaluate** (bool) - 是否返回评估报告。默认值：True。
            - **max_iters** (int) - 选择要变异的种子的最大数量。默认值：10000。
            - **mutate_num_per_seed** (int) - 每个种子的最大变异次数。默认值：20。

        返回：
            - **list** - 模糊测试生成的变异样本。
            - **list** - 变异样本的ground truth标签。
            - **list** - 预测结果。
            - **list** - 变异策略。
            - **dict** - Fuzzer的指标报告。

        异常：
            - **ValueError** - 参数'Coverage'必须是CoverageMetrics的子类。
            - **ValueError** - 初始种子队列为空。
            - **ValueError** - 初始种子队列中的种子不是包含两个元素。

.. py:class:: mindarmour.DPModel(micro_batches=2, norm_bound=1.0, noise_mech=None, clip_mech=None, optimizer=nn.Momentum, **kwargs)

    DPModel用于构建差分隐私训练的模型。

    此类重载 :class:`mindspore.Model`。

    详情请查看： `应用差分隐私机制保护用户隐私 <https://mindspore.cn/mindarmour/docs/zh-CN/r2.0/protect_user_privacy_with_differential_privacy.html#%E5%B7%AE%E5%88%86%E9%9A%90%E7%A7%81>`_。

    参数：
        - **micro_batches** (int) - 从原始批次拆分的小批次数。默认值：2。
        - **norm_bound** (float) - 用于裁剪的约束，如果设置为1，将返回原始数据。默认值：1.0。
        - **noise_mech** (Mechanisms) - 用于生成不同类型的噪音。默认值：None。
        - **clip_mech** (Mechanisms) - 用于更新自适应剪裁。默认值：None。
        - **optimizer** (Cell) - 用于更新差分隐私训练过程中的模型权重值。默认值：nn.Momentum。

    异常：
        - **ValueError** - `optimizer` 值为None。
        - **ValueError** - `optimizer` 不是DPOptimizer，且 `noise_mech` 为None。
        - **ValueError** - `optimizer` 是DPOptimizer，且 `noise_mech` 非None。
        - **ValueError** - `noise_mech` 或DPOptimizer的mech方法是自适应的，而 `clip_mech` 不是None。

.. py:class:: mindarmour.MembershipInference(model, n_jobs=-1)

    成员推理是由Shokri、Stronati、Song和Shmatikov提出的一种用于推测用户隐私数据的灰盒攻击。它需要训练样本的loss或logits结果，隐私是指单个用户的一些敏感属性。

    有关详细信息，请参见：`使用成员推理测试模型安全性 <https://mindspore.cn/mindarmour/docs/zh-CN/r2.0/test_model_security_membership_inference.html>`_。

    参考文献：`Reza Shokri, Marco Stronati, Congzheng Song, Vitaly Shmatikov. Membership Inference Attacks against Machine Learning Models. 2017. <https://arxiv.org/abs/1610.05820v2>`_。

    参数：
        - **model** (Model) - 目标模型。
        - **n_jobs** (int) - 并行运行的任务数量。-1表示使用所有处理器，否则n_jobs的值必须为正整数。

    异常：
        - **TypeError** - 模型的类型不是Mindspore.Model。
        - **TypeError** - `n_jobs` 的类型不是int。
        - **ValueError** - `n_jobs` 的值既不是-1，也不是正整数。

    .. py:method:: eval(dataset_train, dataset_test, metrics)

        评估目标模型的不同隐私。
        评估指标应由metrics规定。

        参数：
            - **dataset_train** (mindspore.dataset) - 目标模型的训练数据集。
            - **dataset_test** (mindspore.dataset) - 目标模型的测试数据集。
            - **metrics** (Union[list, tuple]) - 评估指标。指标的值必须在["precision", "accuracy", "recall"]中。默认值：["precision"]。

        返回：
            - **list** - 每个元素都包含攻击模型的评估指标。

    .. py:method:: train(dataset_train, dataset_test, attack_config)

        根据配置，使用输入数据集训练攻击模型。

        参数：
            - **dataset_train** (mindspore.dataset) - 目标模型的训练数据集。
            - **dataset_test** (mindspore.dataset) - 目标模型的测试集。
            - **attack_config** (Union[list, tuple]) - 攻击模型的参数设置。格式为：

              .. code-block::

                  attack_config = [
                      {"method": "knn", "params": {"n_neighbors": [3, 5, 7]}},
                      {"method": "lr", "params": {"C": np.logspace(-4, 2, 10)}}]

              - 支持的方法有knn、lr、mlp和rf，每个方法的参数必须在可变参数的范围内。参数实现的提示可在下面找到：

                - `KNN <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_
                - `LR <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
                - `RF <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_
                - `MLP <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html>`_

        异常：
            - **KeyError** - `attack_config` 中的配置没有键{"method", "params"}。
            - **NameError** - `attack_config` 中的方法（不区分大小写）不在["lr", "knn", "rf", "mlp"]中。

.. py:class:: mindarmour.ImageInversionAttack(network, input_shape, input_bound, loss_weights=(1, 0.2, 5))

    一种通过还原图像的深层表达来重建图像的攻击方法。

    参考文献：`Aravindh Mahendran, Andrea Vedaldi. Understanding Deep Image Representations by Inverting Them. 2014. <https://arxiv.org/pdf/1412.0035.pdf>`_。

    参数：
        - **network** (Cell) - 网络，用于推断图像的深层特征。
        - **input_shape** (tuple) - 单个网络输入的数据shape，应与给定网络一致。shape的格式应为 :math:`(channel, image\_width, image\_height)`。
        - **input_bound** (Union[tuple, list]) - 原始图像的像素范围，应该像[minimum_pixel, maximum_pixel]或(minimum_pixel, maximum_pixel)。
        - **loss_weights** (Union[list, tuple]) - InversionLoss中三个子损失的权重，可以调整以获得更好的结果。默认值：(1, 0.2, 5)。

    异常：
        - **TypeError** - 网络类型不是Cell。
        - **ValueError** - `input_shape` 的值有非正整数。
        - **ValueError** - `loss_weights` 的值有非正数。

    .. py:method:: evaluate(original_images, inversion_images, labels=None, new_network=None)

        通过三个指标评估还原图像的质量：原始图像和还原图像之间的平均L2距离和SSIM值，以及新模型对还原图像的推理结果在真实标签上的置信度平均值。

        参数：
            - **original_images** (numpy.ndarray) - 原始图像，其shape应为 :math:`(img\_num, channels, img\_width, img\_height)`。
            - **inversion_images** (numpy.ndarray) - 还原图像，其shape应为 :math:`(img\_num, channels, img\_width, img\_height)`。
            - **labels** (numpy.ndarray) - 原始图像的ground truth标签。默认值：None。
            - **new_network** (Cell) - 其结构包含self._network中所有网络，但加载了不同的模型文件。默认值：None。

        返回：
            - **float** - l2距离。
            - **float** - 平均ssim值。
            - **Union** [float, None] - 平均置信度。如果labels或new_network为 None，则该值为None。

    .. py:method:: generate(target_features, iters=100)

        根据 `target_features` 重建图像。

        参数：
            - **target_features** (numpy.ndarray) - 原始图像的深度表示。 `target_features` 的第一个维度应该是img_num。
              需要注意的是，如果img_num等于1，则 `target_features` 的shape应该是 :math:`(1, dim2, dim3, ...)`。
            - **iters** (int) - 逆向攻击的迭代次数，应为正整数。默认值：100。

        返回：
            - **numpy.ndarray** - 重建图像，预计与原始图像相似。

        异常：
            - **TypeError** - target_features的类型不是numpy.ndarray。
            - **ValueError** - `iters` 的有非正整数.

.. py:class:: mindarmour.ConceptDriftCheckTimeSeries(window_size=100, rolling_window=10, step=10, threshold_index=1.5, need_label=False)

    概念漂移检查时间序列（ConceptDriftCheckTimeSeries）用于样本序列分布变化检测。

    有关详细信息，请查看： `实现时序数据概念漂移检测应用 <https://mindspore.cn/mindarmour/docs/zh-CN/r2.0/concept_drift_time_series.html>`_。

    参数：
        - **window_size** (int) - 概念窗口的大小，不小于10。如果给定输入数据，window_size在[10, 1/3*len(input data)]中。如果数据是周期性的，通常window_size等于2-5个周期，例如，对于月/周数据，30/7天的数据量是一个周期。默认值：100。
        - **rolling_window** (int) - 平滑窗口大小，在[1, window_size]中。默认值：10。
        - **step** (int) - 滑动窗口的跳跃长度，在[1, window_size]中。默认值：10。
        - **threshold_index** (float) - 阈值索引，:math:`(-\infty, +\infty)` 。默认值：1.5。
        - **need_label** (bool) - False或True。如果need_label=True，则需要概念漂移标签。默认值：False。

    .. py:method:: concept_check(data)

        在数据序列中查找概念漂移位置。

        参数：
            - **data** (numpy.ndarray) - 输入数据。数据的shape可以是 :math:`(n,1)` 或 :math:`(n,m)`。请注意，每列（m列）是一个数据序列。

        返回：
            - **numpy.ndarray** - 样本序列的概念漂移分数。
            - **float** - 判断概念漂移的阈值。
            - **list** - 概念漂移的位置。
