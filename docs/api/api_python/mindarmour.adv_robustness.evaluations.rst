mindarmour.adv_robustness.evaluations
=====================================

此模块包括各种指标，用于评估攻击或防御的结果。

.. py:class:: mindarmour.adv_robustness.evaluations.AttackEvaluate(inputs, labels, adv_inputs, adv_preds, targeted=False, target_label=None)

    攻击方法的评估指标。

    **参数：**

    - **inputs** (numpy.ndarray) - 原始样本。
    - **labels** (numpy.ndarray) - 原始样本的one-hot格式标签。
    - **adv_inputs** (numpy.ndarray) - 从原始样本生成的对抗样本。
    - **adv_preds** (numpy.ndarray) - 对抗样本的所有输出类的概率。
    - **targeted** (bool) - 如果为True，则为目标攻击。如果为False，则为无目标攻击。默认值：False。
    - **target_label** (numpy.ndarray) - 对抗样本的目标类，是大小为adv_inputs.shape[0]的一维。默认值：None。

    **异常：**

    - **ValueError** - 如果targeted为True时，target_label为None。

    .. py:method:: avg_conf_adv_class()

        计算对抗类的平均置信度（ACAC）。

        **返回：**

        - **float** - 范围在（0,1）之间。值越高，攻击就越成功。

    .. py:method:: avg_conf_true_class()

        计算真类的平均置信度（ACTC）。

        **返回：**

        - **float** - 范围在（0,1）之间。值越低，攻击就越成功。

    .. py:method:: avg_lp_distance()

        计算平均lp距离（lp-dist）。

        **返回：**

        - **float** - 返回所有成功对抗样本的平均'l0'、'l2'或'linf'距离，返回值包括以下情况。
          如果返回值 :math:`>=` 0，则为平均lp距离。值越低，攻击就越成功。
          如果返回值为-1，则没有成功的对抗样本。

    .. py:method:: avg_ssim()

        计算平均结构相似性（ASS）。

        **返回：**

        - **float** - 平均结构相似性。
          如果返回值在（0,1）之间，则值越高，攻击越成功。
          如果返回值为-1，则没有成功的对抗样本。

    .. py:method:: mis_classification_rate()

        计算错误分类率（MR）。

        **返回：**

        - **float** - 范围在（0,1）之间。值越高，攻击就越成功。

    .. py:method:: nte()

        计算噪声容量估计（NTE）。

        参考文献：`Towards Imperceptible and Robust Adversarial Example Attacks against Neural Networks <https://arxiv.org/abs/1801.04693>`_。

        **返回：**

        - **float** - 范围在（0,1）之间。值越高，攻击就越成功。

.. py:class:: mindarmour.adv_robustness.evaluations.BlackDefenseEvaluate(raw_preds, def_preds, raw_query_counts, def_query_counts, raw_query_time, def_query_time, def_detection_counts, true_labels, max_queries)

    反黑盒防御方法的评估指标。

    **参数：**

    - **raw_preds** (numpy.ndarray) - 预测原始模型上某些样本的结果。
    - **def_preds** (numpy.ndarray) - 预测防御模型上某些样本的结果。
    - **raw_query_counts** (numpy.ndarray) - 在原始模型上生成对抗样本的查询数，原始模型是大小为raw_preds.shape[0]的一维。对于良性样本，查询计数必须设置为0。
    - **def_query_counts** (numpy.ndarray) - 在防御模型上生成对抗样本的查询数，原始模型是大小为raw_preds.shape[0]的一维。对于良性样本，查询计数必须设置为0。
    - **raw_query_time** (numpy.ndarray) - 在原始模型上生成对抗样本的总持续时间，该样本是大小为raw_preds.shape[0]的一维。
    - **def_query_time** (numpy.ndarray) - 在防御模型上生成对抗样本的总持续时间，该样本是大小为raw_preds.shape[0]的一维。
    - **def_detection_counts** (numpy.ndarray) - 每次对抗样本生成期间检测到的查询总数，大小为raw_preds.shape[0]的一维。对于良性样本，如果查询被识别为可疑，则将def_detection_counts设置为1，否则将其设置为0。
    - **true_labels** (numpy.ndarray) - 大小为raw_preds.shape[0]的一维真标签。
    - **max_queries** (int) - 攻击预算，最大查询数。

    .. py:method:: asv()

        计算攻击成功率方差（ASV）。

        **返回：**

        - **float** - 值越低，防守就越强。如果num_adv_samples=0，则返回-1。

    .. py:method:: fpr()

        计算基于查询的检测器的假正率（FPR）。

        **返回：**

        - **float** - 值越低，防御的可用性越高。如果num_adv_samples=0，则返回-1。

    .. py:method:: qcv()

        计算查询计数方差（QCV）。

        **返回：**

        - **float** - 值越高，防守就越强。如果num_adv_samples=0，则返回-1。

    .. py:method:: qrv()

        计算良性查询响应时间方差（QRV）。

        **返回：**

        - **float** - 值越低，防御的可用性越高。如果num_adv_samples=0，则返回-1。

.. py:class:: mindarmour.adv_robustness.evaluations.DefenseEvaluate(raw_preds, def_preds, true_labels)

    防御方法的评估指标。

    **参数：**

    - **raw_preds** (numpy.ndarray) - 原始模型上某些样本的预测结果。
    - **def_preds** (numpy.ndarray) - 防御模型上某些样本的预测结果。
    - **true_labels** (numpy.ndarray) - 样本的ground-truth标签，一个大小为ground-truth的一维数组。

    .. py:method:: cav()

        计算分类精度方差（CAV）。

        **返回：**

        - **float** - 值越高，防守就越成功。

    .. py:method:: ccv()

        计算分类置信度方差（CCV）。

        **返回：**

        - **float** - 值越低，防守就越成功。如果返回值== -1，则说明样本数量为0。

    .. py:method:: cos()

        参考文献：`Calculate classification output stability (COS) <https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence>`_。

        **返回：**

        - **float** - 如果返回值>=0，则是有效的防御。值越低，防守越成功。如果返回值== -1, 则说明样本数量为0。

    .. py:method:: crr()

        计算分类校正率（CRR）。

        **返回：**

        - **float** - 值越高，防守就越成功。

    .. py:method:: csr()

        计算分类牺牲比（CSR），越低越好。

        **返回：**

        - **float** - 值越低，防守就越成功。

.. py:class:: mindarmour.adv_robustness.evaluations.RadarMetric(metrics_name, metrics_data, labels, title, scale='hide')

    雷达图，通过多个指标显示模型的鲁棒性。

    **参数：**

    - **metrics_name** (Union[tuple, list]) - 要显示的度量名称数组。每组值对应一条雷达曲线。
    - **metrics_data** (numpy.ndarray) - 多个雷达曲线的每个度量的（归一化）值，如[[0.5, 0.8, ...], [0.2,0.6,...], ...]。
    - **labels** (Union[tuple, list]) - 所有雷达曲线的图例。
    - **title** (str) - 图表的标题。
    - **scale** (str) - 用于调整轴刻度的标量，如'hide'、'norm'、'sparse'、'dense'。默认值：'hide'。

    **异常：**

    - **ValueError** - scale值不在['hide', 'norm', 'sparse', 'dense']中。

    .. py:method:: show()

        显示雷达图。
