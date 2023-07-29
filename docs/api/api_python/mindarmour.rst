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
            - **batch_size** (int) - 一个批次中的样本数。默认值：``64``。

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
            - **is_targeted** (bool) - 对于有目标/无目标攻击，请选择 ``True`` / ``False``。

        返回：
            bool。
            
            - 如果为 ``True``，则输入样本是对抗性的。

            - 如果为 ``False``，则输入样本不是对抗性的。

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
            - **labels** (numpy.ndarray) - 训练数据的标签。默认值：``None``。

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
            - **batch_size** (int) - 一个批次中的样本数。默认值：``32``。
            - **epochs** (int) - epochs的数量。默认值：``5``。

        返回：
            - **numpy.ndarray** - `batch_defense` 操作的损失。

        异常：
            - **ValueError** - `batch_size` 为 ``0``。

    .. py:method:: defense(inputs, labels)

        对输入进行防御操作。

        参数：
            - **inputs** (numpy.ndarray) - 生成对抗样本的原始样本。
            - **labels** (numpy.ndarray) - 输入样本的标签。

        异常：
            - **NotImplementedError** - 抽象方法未实现。
