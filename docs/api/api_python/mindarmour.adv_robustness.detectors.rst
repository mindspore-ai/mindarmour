mindarmour.adv_robustness.detectors
===================================

此模块是用于区分对抗样本和良性样本的检测器方法。

.. py:class:: mindarmour.adv_robustness.detectors.ErrorBasedDetector(auto_encoder, false_positive_rate=0.01, bounds=(0.0, 1.0))

    检测器重建输入样本，测量重建误差，并拒绝重建误差大的样本。

    参考文献： `MagNet: a Two-Pronged Defense against Adversarial Examples, by Dongyu Meng and Hao Chen, at CCS 2017. <https://arxiv.org/abs/1705.09064>`_。

    参数：
        - **auto_encoder** (Model) - 一个（训练过的）自动编码器，对输入图片进行重构。
        - **false_positive_rate** (float) - 检测器的误报率。默认值：0.01。
        - **bounds** (tuple) - (clip_min, clip_max)。默认值：(0.0, 1.0)。

    .. py:method:: detect(inputs)    

        检测输入样本是否具有对抗性。

        参数：
            - **inputs** (numpy.ndarray) - 待判断的可疑样本。

        返回：
            - **list[int]** - 样本是否具有对抗性。如果res[i]=1，则索引为i的输入样本是对抗性的。

    .. py:method:: detect_diff(inputs)    

        检测原始样本和重建样本之间的距离。

        参数：
            - **inputs** (numpy.ndarray) - 输入样本。

        返回：
            - **float** - 重建样本和原始样本之间的距离。

    .. py:method:: fit(inputs, labels=None)    

        查找给定数据集的阈值，以区分对抗样本。

        参数：
            - **inputs** (numpy.ndarray) - 输入样本。
            - **labels** (numpy.ndarray) - 输入样本的标签。默认值：None。

        返回：
            - **float** - 区分对抗样本和良性样本的阈值。

    .. py:method:: set_threshold(threshold)    

        设置阈值。

        参数：
            - **threshold** (float) - 检测阈值。

    .. py:method:: transform(inputs)    

        重建输入样本。

        参数：
            - **inputs** (numpy.ndarray) - 输入样本。

        返回：
            - **numpy.ndarray** - 重建图像。

.. py:class:: mindarmour.adv_robustness.detectors.DivergenceBasedDetector(auto_encoder, model, option='jsd', t=1, bounds=(0.0, 1.0))

    基于发散的检测器学习通过js发散来区分正常样本和对抗样本。

    参考文献： `MagNet: a Two-Pronged Defense against Adversarial Examples, by Dongyu Meng and Hao Chen, at CCS 2017. <https://arxiv.org/abs/1705.09064>`_。

    参数：
        - **auto_encoder** (Model) - 编码器模型。
        - **model** (Model) - 目标模型。
        - **option** (str) - 用于计算发散的方法。默认值：'jsd'。
        - **t** (int) - 用于克服数值问题的温度。默认值：1。
        - **bounds** (tuple) - 数据的上下界。以(clip_min, clip_max)的形式出现。默认值：(0.0, 1.0)。

    .. py:method:: detect_diff(inputs)    

        检测原始样本和重建样本之间的距离。

        距离由JSD计算。

        参数：
            - **inputs** (numpy.ndarray) - 输入样本。

        返回：
            - **float** - 距离。

        异常：
            - **NotImplementedError** - 不支持参数 `option` 。

.. py:class:: mindarmour.adv_robustness.detectors.RegionBasedDetector(model, number_points=10, initial_radius=0.0, max_radius=1.0, search_step=0.01, degrade_limit=0.0, sparse=False)

    基于区域的检测器利用对抗样本靠近分类边界的事实，并通过集成给定示例周围的信息，以检测输入是否为对抗样本。

    参考文献： `Mitigating evasion attacks to deep neural networks via region-based classification <https://arxiv.org/abs/1709.05583>`_。

    参数：
        - **model** (Model) - 目标模型。
        - **number_points** (int) - 从原始样本的超立方体生成的样本数。默认值：10。
        - **initial_radius** (float) - 超立方体的初始半径。默认值：0.0。
        - **max_radius** (float) - 超立方体的最大半径。默认值：1.0。
        - **search_step** (float) - 半径搜索增量。默认值：0.01。
        - **degrade_limit** (float) - 分类精度的可接受下降。默认值：0.0。
        - **sparse** (bool) - 如果为True，则输入标签为稀疏编码。如果为False，则输入标签为onehot编码。默认值：False。

    .. py:method:: detect(inputs)    

        判断输入样本是否具有对抗性。

        参数：
            - **inputs** (numpy.ndarray) - 待判断的可疑样本。

        返回：
            - **list[int]** - 样本是否具有对抗性。如果res[i]=1，则索引为i的输入样本是对抗性的。

    .. py:method:: detect_diff(inputs)    

        返回原始预测结果和基于区域的预测结果。

        参数：
            - **inputs** (numpy.ndarray) - 输入样本。

        返回：
            - **numpy.ndarray** - 输入样本的原始预测结果和基于区域的预测结果。

    .. py:method:: fit(inputs, labels=None)    

        训练检测器来决定最佳半径。

        参数：
            - **inputs** (numpy.ndarray) - 良性样本。
            - **labels** (numpy.ndarray) - 输入样本的ground truth标签。默认值：None。

        返回：
            - **float** - 最佳半径。

    .. py:method:: set_radius(radius)    

        设置半径。

        参数：
            - **radius** (float) - 区域的半径。

    .. py:method:: transform(inputs)    

        为输入样本生成超级立方体。

        参数：
            - **inputs** (numpy.ndarray) - 输入样本。

        返回：
            - **numpy.ndarray** - 每个样本对应的超立方体。

.. py:class:: mindarmour.adv_robustness.detectors.SpatialSmoothing(model, ksize=3, is_local_smooth=True, metric='l1', false_positive_ratio=0.05)

    基于空间平滑的检测方法。
    使用高斯滤波、中值滤波和均值滤波，模糊原始图像。当模型在样本模糊前后的预测值之间有很大的阈值差异时，将其判断为对抗样本。

    参数：
        - **model** (Model) - 目标模型。
        - **ksize** (int) - 平滑窗口大小。默认值：3。
        - **is_local_smooth** (bool) - 如果为True，则触发局部平滑。如果为False，则无局部平滑。默认值：True。
        - **metric** (str) - 距离方法。默认值：'l1'。
        - **false_positive_ratio** (float) - 良性样本上的假正率。默认值：0.05。

    .. py:method:: detect(inputs)    

        检测输入样本是否为对抗样本。

        参数：
            - **inputs** (numpy.ndarray) - 待判断的可疑样本。

        返回：
            - **list[int]** - 样本是否具有对抗性。如果res[i]=1，则索引为i的输入样本是对抗样本。

    .. py:method:: detect_diff(inputs)    

        返回输入样本与其平滑对应样本之间的原始距离值（在应用阈值之前）。

        参数：
            - **inputs** (numpy.ndarray) - 待判断的可疑样本。

        返回：
            - **float** - 距离。

    .. py:method:: fit(inputs, labels=None)    

        训练检测器来决定阈值。适当的阈值能够确保良性样本上的实际假正率小于给定值。

        参数：
            - **inputs** (numpy.ndarray) - 良性样本。
            - **labels** (numpy.ndarray) - 默认None。

        返回：
            - **float** - 阈值，大于该距离的距离报告为正，即对抗性。

    .. py:method:: set_threshold(threshold)    

        设置阈值。

        参数：
            - **threshold** (float) - 检测阈值。

.. py:class:: mindarmour.adv_robustness.detectors.EnsembleDetector(detectors, policy='vote')

    集合检测器，通过检测器列表从输入样本中检测对抗样本。

    参数：
        - **detectors** (Union[tuple, list]) - 检测器方法列表。
        - **policy** (str) - 决策策略，取值可为'vote'、'all'、'any'。默认值：'vote'

    .. py:method:: detect(inputs)    

        从输入样本中检测对抗性示例。

        参数：
            - **inputs** (numpy.ndarray) - 输入样本。

        返回：
            - **list[int]** - 样本是否具有对抗性。如果res[i]=1，则索引为i的输入样本是对抗样本。

        异常：
            - **ValueError** - 不支持策略。

    .. py:method:: detect_diff(inputs)    

        此方法在此类中不可用。

        参数：
            - **inputs** (Union[numpy.ndarray, list, tuple]) - 用于创建对抗样本。

        异常：
            - **NotImplementedError** - 此函数在集成中不可用。

    .. py:method:: fit(inputs, labels=None)    

        像机器学习模型一样拟合检测器。此方法在此类中不可用。

        参数：
            - **inputs** (numpy.ndarray) - 计算阈值的数据。
            - **labels** (numpy.ndarray) - 数据的标签。默认值：None。

        异常：
            - **NotImplementedError** - 此函数在集成中不可用。

    .. py:method:: transform(inputs)    

        过滤输入样本中的对抗性噪声。
        此方法在此类中不可用。

        参数：
            - **inputs** (Union[numpy.ndarray, list, tuple]) - 用于创建对抗样本。

        异常：
            - **NotImplementedError** - 此函数在集成中不可用。

.. py:class:: mindarmour.adv_robustness.detectors.SimilarityDetector(trans_model, max_k_neighbor=1000, chunk_size=1000, max_buffer_size=10000, tuning=False, fpr=0.001)

    检测器测量相邻查询之间的相似性，并拒绝与以前的查询非常相似的查询。

    参考文献： `Stateful Detection of Black-Box Adversarial Attacks by Steven Chen, Nicholas Carlini, and David Wagner. at arxiv 2019 <https://arxiv.org/abs/1907.05587>`_。

    参数：
        - **trans_model** (Model) - 一个MindSpore模型，将输入数据编码为低维向量。
        - **max_k_neighbor** (int) - 最近邻的最大数量。默认值：1000。
        - **chunk_size** (int) - 缓冲区大小。默认值：1000。
        - **max_buffer_size** (int) - 最大缓冲区大小。默认值：10000。
        - **tuning** (bool) - 计算k个最近邻的平均距离。

          - 如果'tuning'为true，k= `max_k_neighbor` 。
          - 如果为False，k=1,..., `max_k_neighbor` 。默认值：False。

        - **fpr** (float) - 合法查询序列上的误报率。默认值：0.001

    .. py:method:: clear_buffer()    

        清除缓冲区内存。

    .. py:method:: detect(inputs)    

        处理查询以检测黑盒攻击。

        参数：
            - **inputs** (numpy.ndarray) - 查询序列。

        异常：
            - **ValueError** - 阈值或set_threshold方法中 `num_of_neighbors` 参数不可用。

    .. py:method:: detect_diff(inputs)    

        从输入样本中检测对抗样本，如常见机器学习模型中的predict_proba函数。

        参数：
            - **inputs** (Union[numpy.ndarray, list, tuple]) - 用于创建对抗样本。

        异常：
            - **NotImplementedError** - 此函数在 `SimilarityDetector` 类（class）中不可用。

    .. py:method:: fit(inputs, labels=None)    

        处理输入训练数据以计算阈值。
        适当的阈值应确保假正率低于给定值。

        参数：
            - **inputs** (numpy.ndarray) - 用于计算阈值的训练数据。
            - **labels** (numpy.ndarray) - 训练数据的标签。

        返回：
            - **list[int]** - 最近邻的数量。

            - **list[float]** - 不同k的阈值。

        异常：
            - **ValueError** - 训练数据个数小于 `max_k_neighbor`。

    .. py:method:: get_detected_queries()    

        获取检测到的查询的索引。

        返回：
            - **list[int]** - 检测到的恶意查询的序列号。

    .. py:method:: get_detection_interval()    

        获取相邻检测之间的间隔。

        返回：
            - **list[int]** - 相邻检测之间的查询数。

    .. py:method:: set_threshold(num_of_neighbors, threshold)    

        设置参数num_of_neighbors和threshold。

        参数：
            - **num_of_neighbors** (int) - 最近邻的数量。
            - **threshold** (float) - 检测阈值。

    .. py:method:: transform(inputs)    

        过滤输入样本中的对抗性噪声。

        参数：
            - **inputs** (Union[numpy.ndarray, list, tuple]) - 用于创建对抗样本。

        异常：
            - **NotImplementedError** - 此函数在 `SimilarityDetector` 类（class）中不可用。
