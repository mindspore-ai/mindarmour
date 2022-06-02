mindarmour.adv_robustness.defenses
==================================

该模块包括经典的防御算法，用于防御对抗样本，增强模型的安全性和可信性。

.. py:class:: mindarmour.adv_robustness.defenses.AdversarialDefense(network, loss_fn=None, optimizer=None)

    使用给定的对抗样本进行对抗训练。

    **参数：**

    - **network** (Cell) - 要防御的MindSpore网络。
    - **loss_fn** (Functions) - 损失函数。默认值：None。
    - **optimizer** (Cell) - 用于训练网络的优化器。默认值：None。

    .. py:method:: defense(inputs, labels)

        通过使用输入样本进行训练来增强模型。

        **参数：**

        - **inputs** (numpy.ndarray) - 输入样本。
        - **labels** (numpy.ndarray) - 输入样本的标签。

        **返回：**

        - **numpy.ndarray** - 防御操作的损失。

.. py:class:: mindarmour.adv_robustness.defenses.AdversarialDefenseWithAttacks(network, attacks, loss_fn=None, optimizer=None, bounds=(0.0, 1.0), replace_ratio=0.5)

    利用特定的攻击方法和给定的对抗例子进行对抗训练，以增强模型的鲁棒性。

    **参数：**

    - **network** (Cell) - 要防御的MindSpore网络。
    - **attacks** (list[Attack]) - 攻击方法列表。
    - **loss_fn** (Functions) - 损失函数。默认值：None。
    - **optimizer** (Cell) - 用于训练网络的优化器。默认值：None。
    - **bounds** (tuple) - 数据的上下界。以(clip_min, clip_max)的形式出现。默认值：(0.0, 1.0)。
    - **replace_ratio** (float) - 用对抗性替换原始样本的比率，必须在0到1之间。默认值：0.5。

    **异常：**

    - **ValueError** - 替换比率不在0和1之间。

    .. py:method:: defense(inputs, labels)

        通过使用从输入样本生成的对抗样本进行训练来增强模型。

        **参数：**

        - **inputs** (numpy.ndarray) - 输入样本。
        - **labels** (numpy.ndarray) - 输入样本的标签。

        **返回：**

        - **numpy.ndarray** - 对抗性防御操作的损失。

.. py:class:: mindarmour.adv_robustness.defenses.NaturalAdversarialDefense(network, loss_fn=None, optimizer=None, bounds=(0.0, 1.0), replace_ratio=0.5, eps=0.1)

    基于FGSM的对抗性训练。

    参考文献：`A. Kurakin, et al., "Adversarial machine learning at scale," in ICLR, 2017. <https://arxiv.org/abs/1611.01236>`_ 。

    **参数：**

    - **network** (Cell) - 要防御的MindSpore网络。
    - **loss_fn** (Functions) - 损失函数。默认值：None。
    - **optimizer** (Cell)：用于训练网络的优化器。默认值：None。
    - **bounds** (tuple) - 数据的上下界。以(clip_min, clip_max)的形式出现。默认值：(0.0, 1.0)。
    - **replace_ratio** (float) - 用对抗样本替换原始样本的比率。默认值：0.5。
    - **eps** (float) - 攻击方法（FGSM）的步长。默认值：0.1。

.. py:class:: mindarmour.adv_robustness.defenses.ProjectedAdversarialDefense(network, loss_fn=None, optimizer=None, bounds=(0.0, 1.0), replace_ratio=0.5, eps=0.3, eps_iter=0.1, nb_iter=5, norm_level='inf')

    基于PGD的对抗性训练。

    参考文献：`A. Madry, et al., "Towards deep learning models resistant to adversarial attacks," in ICLR, 2018. <https://arxiv.org/abs/1611.01236>`_ 。

    **参数：**

    - **network** (Cell) - 要防御的MindSpore网络。
    - **loss_fn** (Functions) - 损失函数。默认值：None。
    - **optimizer** (Cell) - 用于训练网络的优化器。默认值：None。
    - **bounds** (tuple) - 输入数据的上下界。以(clip_min, clip_max)的形式出现。默认值：(0.0, 1.0)。
    - **replace_ratio** (float) - 用对抗样本替换原始样本的比率。默认值：0.5。
    - **eps** (float) - PGD攻击参数epsilon。默认值：0.3。
    - **eps_iter** (int) - PGD攻击参数，内环epsilon。默认值：0.1。
    - **nb_iter** (int) - PGD攻击参数，迭代次数。默认值：5。
    - **norm_level** (Union[int, char, numpy.inf]) - 范数类型。可选值：1、2、np.inf、'l1'、'l2'、'np.inf' 或 'inf'。默认值：'inf'。

.. py:class:: mindarmour.adv_robustness.defenses.EnsembleAdversarialDefense(network, attacks, loss_fn=None, optimizer=None, bounds=(0.0, 1.0), replace_ratio=0.5)

    使用特定攻击方法列表和给定的对抗样本进行对抗训练，以增强模型的鲁棒性。

    **参数：**

    - **network** (Cell) - 要防御的MindSpore网络。
    - **attacks** (list[Attack]) - 攻击方法列表。
    - **loss_fn** (Functions) - 损失函数。默认值：None。
    - **optimizer** (Cell) - 用于训练网络的优化器。默认值：None。
    - **bounds** (tuple) - 数据的上下界。以(clip_min, clip_max)的形式出现。默认值：(0.0, 1.0)。
    - **replace_ratio** (float) - 用对抗性替换原始样本的比率，必须在0到1之间。默认值：0.5。

    **异常：**

    - **ValueError** - `bounds` 不在0和1之间。
