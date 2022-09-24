mindarmour.reliability
======================

MindArmour的可靠性方法。

.. py:class:: mindarmour.reliability.FaultInjector(model, fi_type=None, fi_mode=None, fi_size=None)

    故障注入模块模拟深度神经网络的各种故障场景，并评估模型的性能和可靠性。

    详情请查看 `实现模型故障注入评估模型容错性 <https://mindspore.cn/mindarmour/docs/zh-CN/r1.9/fault_injection.html>`_。

    参数：
        - **model** (Model) - 需要评估模型。
        - **fi_type** (list) - 故障注入的类型，包括'bitflips_random'（随机翻转）、'bitflips_designated'（翻转关键位）、'random'、'zeros'、'nan'、'inf'、'anti_activation'、'precision_loss'等。
        - **fi_mode** (list) - 故障注入的模式。可选值：'single_layer'，'all_layer'。
        - **fi_size** (list) - 故障注入的次数，表示需要注入多少值。

    .. py:method:: kick_off(ds_data, ds_label, iter_times=100)

        启动故障注入并返回最终结果。

        参数：
            - **ds_data** (np.ndarray) - 输入测试数据。评估基于这些数据。
            - **ds_label** (np.ndarray) - 数据的标签，对应于数据。
            - **iter_times** (int) - 评估数，这将决定批处理大小。

        返回：
            - **list** - 故障注入的结果。

    .. py:method:: metrics()

        最终结果的指标。

        返回：
            - **list** - 结果总结。

.. py:class:: mindarmour.reliability.ConceptDriftCheckTimeSeries(window_size=100, rolling_window=10, step=10, threshold_index=1.5, need_label=False)

    概念漂移检查时间序列（ConceptDriftCheckTimeSeries）用于样本序列分布变化检测。

    有关详细信息，请查看 `实现时序数据概念漂移检测应用
    <https://mindspore.cn/mindarmour/docs/zh-CN/r1.9/concept_drift_time_series.html>`_。

    参数：
        - **window_size** (int) - 概念窗口的大小，不小于10。如果给定输入数据， `window_size` 在[10, 1/3*len( `data` )]中。
          如果数据是周期性的，通常 `window_size` 等于2-5个周期。例如，对于月/周数据，30/7天的数据量是一个周期。默认值：100。
        - **rolling_window** (int) - 平滑窗口大小，在[1, `window_size` ]中。默认值：10。
        - **step** (int) - 滑动窗口的跳跃长度，在[1, `window_size` ]中。默认值：10。
        - **threshold_index** (float) - 阈值索引，:math:`(-\infty, +\infty)` 。默认值：1.5。
        - **need_label** (bool) - False或True。如果 `need_label` =True，则需要概念漂移标签。默认值：False。

    .. py:method:: concept_check(data)

        在数据序列中查找概念漂移位置。

        参数：
            - **data** (numpy.ndarray) - 输入数据。数据的shape可以是(n,1)或(n,m)。
              请注意，每列（m列）是一个数据序列。

        返回：
            - **numpy.ndarray** - 样本序列的概念漂移分数。
            - **float** - 判断概念漂移的阈值。
            - **list** - 概念漂移的位置。

.. py:class:: mindarmour.reliability.OodDetector(model, ds_train)

    分布外检测器的抽象类。

    参数：
        - **model** (Model) - 训练模型。
        - **ds_train** (numpy.ndarray) - 训练数据集。

    .. py:method:: get_optimal_threshold(label, ds_eval)

        获取最佳阈值。尝试找到一个最佳阈值来检测OOD样本。最佳阈值由标记的数据集 `ds_eval` 计算。

        参数：
            - **label** (numpy.ndarray) - 区分图像是否为分布内或分布外的标签。
            - **ds_eval** (numpy.ndarray) - 帮助查找阈值的测试数据集。

        返回：
            - **float** - 最佳阈值。

    .. py:method:: ood_predict(threshold, ds_test)

        分布外（out-of-distribution，OOD）检测。此函数的目的是检测被视为 `ds_test` 的图像是否为OOD样本。如果一张图像的预测分数大于 `threshold` ，则该图像为分布外。

        参数：
            - **threshold** (float) - 判断ood数据的阈值。可以根据经验设置值，也可以使用函数get_optimal_threshold。
            - **ds_test** (numpy.ndarray) - 测试数据集。

        返回：
            - **numpy.ndarray** - 检测结果。0表示数据不是ood，1表示数据是ood。

.. py:class:: mindarmour.reliability.OodDetectorFeatureCluster(model, ds_train, n_cluster, layer)

    训练OOD检测器。提取训练数据特征，得到聚类中心。测试数据特征与聚类中心之间的距离确定图像是否为分布外（OOD）图像。

    有关详细信息，请查看 `实现图像数据概念漂移检测应用 <https://mindspore.cn/mindarmour/docs/zh-CN/r1.9/concept_drift_images.html>`_。

    参数：
        - **model** (Model) - 训练模型。
        - **ds_train** (numpy.ndarray) - 训练数据集。
        - **n_cluster** (int) - 聚类数量。取值属于[2,100]。
          通常，n_cluster等于训练数据集的类号。如果OOD检测器在测试数据集中性能较差，我们可以适当增加n_cluster的值。
        - **layer** (str) - 特征层的名称。layer (str)由'name[:Tensor]'表示，其中'name'由用户在训练模型时给出。
          请查看有关如何在'README.md'中命名模型层的更多详细信息。

    .. py:method:: get_optimal_threshold(label, ds_eval)

        获取最佳阈值。尝试找到一个最佳阈值来检测OOD样本。最佳阈值由标记的数据集 `ds_eval` 计算。

        参数：
            - **label** (numpy.ndarray) - 区分图像是否为分布内或分布外的标签。
            - **ds_eval** (numpy.ndarray) - 帮助查找阈值的测试数据集。

        返回：
            - **float** - 最佳阈值。

    .. py:method:: ood_predict(threshold, ds_test)

        分布外（out-of-distribution，OOD）检测。此函数的目的是检测 `ds_test` 中的图像是否为OOD样本。如果一张图像的预测分数大于 `threshold` ，则该图像为分布外。

        参数：
            - **threshold** (float) - 判断ood数据的阈值。可以根据经验设置值，也可以使用函数get_optimal_threshold。
            - **ds_test** (numpy.ndarray) - 测试数据集。

        返回：
            - **numpy.ndarray** - 检测结果。0表示数据不是ood，1表示数据是ood。
