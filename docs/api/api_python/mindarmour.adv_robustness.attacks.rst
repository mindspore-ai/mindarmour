mindarmour.adv_robustness.attacks
=================================

本模块包括经典的黑盒和白盒攻击算法，以制作对抗样本。

.. py:class:: mindarmour.adv_robustness.attacks.FastGradientMethod(network, eps=0.07, alpha=None, bounds=(0.0, 1.0), norm_level=2, is_targeted=False, loss_fn=None)

    基于梯度计算的单步攻击，扰动的范数包括 'L1'、'L2'和'Linf'。

    参考文献：`I. J. Goodfellow, J. Shlens, and C. Szegedy, "Explaining and harnessing adversarial examples," in ICLR, 2015. <https://arxiv.org/abs/1412.6572>`_。

    参数：
        - **network** (Cell) - 目标模型。
        - **eps** (float) - 攻击产生的单步对抗扰动占数据范围的比例。默认值：0.07。
        - **alpha** (Union[float, None]) - 单步随机扰动与数据范围的比例。默认值：None。
        - **bounds** (tuple) - 数据的上下界，表示数据范围。以(数据最小值, 数据最大值)的形式出现。默认值：(0.0, 1.0)。
        - **norm_level** (Union[int, str, numpy.inf]) - 范数类型。
          可取值：numpy.inf、1、2、'1'、'2'、'l1'、'l2'、'np.inf'、'inf'、'linf'。默认值：2。
        - **is_targeted** (bool) - 如果为True，则为目标攻击。如果为False，则为无目标攻击。默认值：False。
        - **loss_fn** (Union[loss, None]) - 用于优化的损失函数。如果为None，则输入网络已配备损失函数。默认值：None。

.. py:class:: mindarmour.adv_robustness.attacks.RandomFastGradientMethod(network, eps=0.07, alpha=0.035, bounds=(0.0, 1.0), norm_level=2, is_targeted=False, loss_fn=None)

    使用随机扰动的快速梯度法（Fast Gradient Method）。
    基于梯度计算的单步攻击，其对抗性噪声是根据输入的梯度生成的，然后加入随机扰动，从而生成对抗样本。

    参考文献：`Florian Tramer, Alexey Kurakin, Nicolas Papernot, "Ensemble adversarial training: Attacks and defenses" in ICLR, 2018 <https://arxiv.org/abs/1705.07204>`_。

    参数：
        - **network** (Cell) - 目标模型。
        - **eps** (float) - 攻击产生的单步对抗扰动占数据范围的比例。默认值：0.07。
        - **alpha** (float) - 单步随机扰动与数据范围的比例。默认值：0.035。
        - **bounds** (tuple) - 数据的上下界，表示数据范围。以(数据最小值，数据最大值)的形式出现。默认值：(0.0, 1.0)。
        - **norm_level** (Union[int, str, numpy.inf]) - 范数类型。
          可取值：numpy.inf、1、2、'1'、'2'、'l1'、'l2'、'np.inf'、'inf'、'linf'。默认值：2。
        - **is_targeted** (bool) - 如果为True，则为目标攻击。如果为False，则为无目标攻击。默认值：False。
        - **loss_fn** (Union[loss, None]) - 用于优化的损失函数。如果为None，则输入网络已配备损失函数。默认值：None。

    异常：
        - **ValueError** - `eps` 小于 `alpha` 。

.. py:class:: mindarmour.adv_robustness.attacks.FastGradientSignMethod(network, eps=0.07, alpha=None, bounds=(0.0, 1.0), is_targeted=False, loss_fn=None)

    快速梯度下降法（Fast Gradient Sign Method）攻击计算输入数据的梯度，然后使用梯度的符号创建对抗性噪声。

    参考文献：`Ian J. Goodfellow, J. Shlens, and C. Szegedy, "Explaining and harnessing adversarial examples," in ICLR, 2015 <https://arxiv.org/abs/1412.6572>`_。

    参数：
        - **network** (Cell) - 目标模型。
        - **eps** (float) - 攻击产生的单步对抗扰动占数据范围的比例。默认值：0.07。
        - **alpha** (Union[float, None]) - 单步随机扰动与数据范围的比例。默认值：None。
        - **bounds** (tuple) - 数据的上下界，表示数据范围。以(数据最小值，数据最大值)的形式出现。默认值：(0.0, 1.0)。
        - **is_targeted** (bool) - 如果为True，则为目标攻击。如果为False，则为无目标攻击。默认值：False。
        - **loss_fn** (Union[Loss, None]) - 用于优化的损失函数。如果为None，则输入网络已配备损失函数。默认值：None。

.. py:class:: mindarmour.adv_robustness.attacks.RandomFastGradientSignMethod(network, eps=0.07, alpha=0.035, bounds=(0.0, 1.0), is_targeted=False, loss_fn=None)

    快速梯度下降法（Fast Gradient Sign Method）使用随机扰动。
    随机快速梯度符号法（Random Fast Gradient Sign Method）攻击计算输入数据的梯度，然后使用带有随机扰动的梯度符号来创建对抗性噪声。

    参考文献：`F. Tramer, et al., "Ensemble adversarial training: Attacks and defenses," in ICLR, 2018 <https://arxiv.org/abs/1705.07204>`_。

    参数：
        - **network** (Cell) - 目标模型。
        - **eps** (float) - 攻击产生的单步对抗扰动占数据范围的比例。默认值：0.07。
        - **alpha** (float) - 单步随机扰动与数据范围的比例。默认值：0.005。
        - **bounds** (tuple) - 数据的上下界，表示数据范围。以(数据最小值，数据最大值)的形式出现。默认值：(0.0, 1.0)。
        - **is_targeted** (bool) - 如果为True，则为目标攻击。如果为False，则为无目标攻击。默认值：False。
        - **loss_fn** (Union[Loss, None]) - 用于优化的损失函数。如果为None，则输入网络已配备损失函数。默认值：None。

    异常：
        - **ValueError** - `eps` 小于 `alpha` 。

.. py:class:: mindarmour.adv_robustness.attacks.LeastLikelyClassMethod(network, eps=0.07, alpha=None, bounds=(0.0, 1.0), loss_fn=None)

    单步最不可能类方法（Single Step Least-Likely Class Method）是FGSM的变体，它以最不可能类为目标，以生成对抗样本。

    参考文献：`F. Tramer, et al., "Ensemble adversarial training: Attacks and defenses," in ICLR, 2018 <https://arxiv.org/abs/1705.07204>`_。

    参数：
        - **network** (Cell) - 目标模型。
        - **eps** (float) - 攻击产生的单步对抗扰动占数据范围的比例。默认值：0.07。
        - **alpha** (Union[float, None]) - 单步随机扰动与数据范围的比例。默认值：None。
        - **bounds** (tuple) - 数据的上下界，表示数据范围。以(数据最小值，数据最大值)的形式出现。默认值：(0.0, 1.0)。
        - **loss_fn** (Union[Loss, None]) - 用于优化的损失函数。如果为None，则输入网络已配备损失函数。默认值：None。

.. py:class:: mindarmour.adv_robustness.attacks.RandomLeastLikelyClassMethod(network, eps=0.07, alpha=0.035, bounds=(0.0, 1.0), loss_fn=None)

    随机最不可能类攻击方法：以置信度最小类别对应的梯度加一个随机扰动为攻击方向。

    具有随机扰动的单步最不可能类方法（Single Step Least-Likely Class Method）是随机FGSM的变体，它以最不可能类为目标，以生成对抗样本。

    参考文献：`F. Tramer, et al., "Ensemble adversarial training: Attacks and defenses," in ICLR, 2018 <https://arxiv.org/abs/1705.07204>`_。

    参数：
        - **network** (Cell) - 目标模型。
        - **eps** (float) - 攻击产生的单步对抗扰动占数据范围的比例。默认值：0.07。
        - **alpha** (float) - 单步随机扰动与数据范围的比例。默认值：0.005。
        - **bounds** (tuple) - 数据的上下界，表示数据范围。以(数据最小值，数据最大值)的形式出现。默认值：(0.0, 1.0)。
        - **loss_fn** (Union[Loss, None]) - 用于优化的损失函数。如果为None，则输入网络已配备损失函数。默认值：None。

    异常：
        - **ValueError** - `eps` 小于 `alpha` 。

.. py:class:: mindarmour.adv_robustness.attacks.IterativeGradientMethod(network, eps=0.3, eps_iter=0.1, bounds=(0.0, 1.0), nb_iter=5, loss_fn=None)

    所有基于迭代梯度的攻击的抽象基类。

    参数：
        - **network** (Cell) - 目标模型。
        - **eps** (float) - 攻击产生的对抗性扰动占数据范围的比例。默认值：0.3。
        - **eps_iter** (float) - 攻击产生的单步对抗扰动占数据范围的比例。默认值：0.1。
        - **bounds** (tuple) - 数据的上下界，表示数据范围。以(数据最小值，数据最大值)的形式出现。默认值：(0.0, 1.0)。
        - **nb_iter** (int) - 迭代次数。默认值：5。
        - **loss_fn** (Union[Loss, None]) - 用于优化的损失函数。如果为None，则输入网络已配备损失函数。默认值：None。

    .. py:method:: generate(inputs, labels)

        根据输入样本和原始/目标标签生成对抗样本。

        参数：
            - **inputs** (Union[numpy.ndarray, tuple]) - 良性输入样本，用于创建对抗样本。
            - **labels** (Union[numpy.ndarray, tuple]) - 原始/目标标签。若每个输入有多个标签，将它包装在元组中。

        异常：
            - **NotImplementedError** - 此函数在迭代梯度方法中不可用。

.. py:class:: mindarmour.adv_robustness.attacks.BasicIterativeMethod(network, eps=0.3, eps_iter=0.1, bounds=(0.0, 1.0), is_targeted=False, nb_iter=5, loss_fn=None)

    基本迭代法（Basic Iterative Method）攻击，一种生成对抗示例的迭代FGSM方法。

    参考文献：`A. Kurakin, I. Goodfellow, and S. Bengio, "Adversarial examples in the physical world," in ICLR, 2017 <https://arxiv.org/abs/1607.02533>`_。

    参数：
        - **network** (Cell) - 目标模型。
        - **eps** (float) - 攻击产生的对抗性扰动占数据范围的比例。默认值：0.3。
        - **eps_iter** (float) - 攻击产生的单步对抗扰动占数据范围的比例。默认值：0.1。
        - **bounds** (tuple) - 数据的上下界，表示数据范围。以(数据最小值，数据最大值)的形式出现。默认值：(0.0, 1.0)。
        - **is_targeted** (bool) - 如果为True，则为目标攻击。如果为False，则为无目标攻击。默认值：False。
        - **nb_iter** (int) - 迭代次数。默认值：5。
        - **loss_fn** (Union[Loss, None]) - 用于优化的损失函数。如果为None，则输入网络已配备损失函数。默认值：None。

    .. py:method:: generate(inputs, labels)    

        使用迭代FGSM方法生成对抗样本。

        参数：
            - **inputs** (Union[numpy.ndarray, tuple]) - 良性输入样本，用于创建对抗样本。
            - **labels** (Union[numpy.ndarray, tuple]) - 原始/目标标签。若每个输入有多个标签，将它包装在元组中。

        返回：
            - **numpy.ndarray** - 生成的对抗样本。

.. py:class:: mindarmour.adv_robustness.attacks.MomentumIterativeMethod(network, eps=0.3, eps_iter=0.1, bounds=(0.0, 1.0), is_targeted=False, nb_iter=5, decay_factor=1.0, norm_level='inf', loss_fn=None)

    动量迭代法（Momentum Iterative Method）攻击，通过在迭代中积累损失函数的梯度方向上的速度矢量，加速梯度下降算法，如FGSM、FGM和LLCM，从而生成对抗样本。

    参考文献：`Y. Dong, et al., "Boosting adversarial attacks with momentum," arXiv:1710.06081, 2017 <https://arxiv.org/abs/1710.06081>`_。

    参数：
        - **network** (Cell) - 目标模型。
        - **eps** (float) - 攻击产生的对抗性扰动占数据范围的比例。默认值：0.3。
        - **eps_iter** (float) - 攻击产生的单步对抗扰动占数据范围的比例。默认值：0.1。
        - **bounds** (tuple) - 数据的上下界，表示数据范围。
          以(数据最小值，数据最大值)的形式出现。默认值：(0.0, 1.0)。
        - **is_targeted** (bool) - 如果为True，则为目标攻击。如果为False，则为无目标攻击。默认值：False。
        - **nb_iter** (int) - 迭代次数。默认值：5。
        - **decay_factor** (float) - 迭代中的衰变因子。默认值：1.0。
        - **norm_level** (Union[int, str, numpy.inf]) - 范数类型。
          可取值：numpy.inf、1、2、'1'、'2'、'l1'、'l2'、'np.inf'、'inf'、'linf'。默认值：numpy.inf。
        - **loss_fn** (Union[Loss, None]) - 用于优化的损失函数。如果为None，则输入网络已配备损失函数。默认值：None。

    .. py:method:: generate(inputs, labels)    

        根据输入数据和原始/目标标签生成对抗样本。

        参数：
            - **inputs** (Union[numpy.ndarray, tuple]) - 良性输入样本，用于创建对抗样本。
            - **labels** (Union[numpy.ndarray, tuple]) - 原始/目标标签。若每个输入有多个标签，将它包装在元组中。

        返回：
            - **numpy.ndarray** - 生成的对抗样本。

.. py:class:: mindarmour.adv_robustness.attacks.ProjectedGradientDescent(network, eps=0.3, eps_iter=0.1, bounds=(0.0, 1.0), is_targeted=False, nb_iter=5, norm_level='inf', loss_fn=None)

    投影梯度下降（Projected Gradient Descent）攻击是基本迭代法的变体，在这种方法中，每次迭代之后，扰动被投影在指定半径的p范数球上（除了剪切对抗样本的值，使其位于允许的数据范围内）。这是Madry等人提出的用于对抗性训练的攻击。

    参考文献：`A. Madry, et al., "Towards deep learning models resistant to adversarial attacks," in ICLR, 2018 <https://arxiv.org/abs/1706.06083>`_。

    参数：
        - **network** (Cell) - 目标模型。
        - **eps** (float) - 攻击产生的对抗性扰动占数据范围的比例。默认值：0.3。
        - **eps_iter** (float) - 攻击产生的单步对抗扰动占数据范围的比例。默认值：0.1。
        - **bounds** (tuple) - 数据的上下界，表示数据范围。以(数据最小值，数据最大值)的形式出现。默认值：(0.0, 1.0)。
        - **is_targeted** (bool) - 如果为True，则为目标攻击。如果为False，则为无目标攻击。默认值：False。
        - **nb_iter** (int) - 迭代次数。默认值：5。
        - **norm_level** (Union[int, str, numpy.inf]) - 范数类型。
          可取值：numpy.inf、1、2、'1'、'2'、'l1'、'l2'、'np.inf'、'inf'、'linf'。默认值：'numpy.inf'。
        - **loss_fn** (Union[Loss, None]) - 用于优化的损失函数。如果为None，则输入网络已配备损失函数。默认值：None。

    .. py:method:: generate(inputs, labels)

        基于BIM方法迭代生成对抗样本。通过带有参数norm_level的投影方法归一化扰动。

        参数：
            - **inputs** (Union[numpy.ndarray, tuple]) - 良性输入样本，用于创建对抗样本。
            - **labels** (Union[numpy.ndarray, tuple]) - 原始/目标标签。若每个输入有多个标签，将它包装在元组中。

        返回：
            - **numpy.ndarray** - 生成的对抗样本。

.. py:class:: mindarmour.adv_robustness.attacks.DiverseInputIterativeMethod(network, eps=0.3, bounds=(0.0, 1.0), is_targeted=False, prob=0.5, loss_fn=None)

    多样性输入迭代法（Diverse Input Iterative Method）攻击遵循基本迭代法，并在每次迭代时对输入数据应用随机转换。对输入数据的这种转换可以提高对抗样本的可转移性。

    参考文献：`Xie, Cihang and Zhang, et al., "Improving Transferability of Adversarial Examples With Input Diversity," in CVPR, 2019 <https://arxiv.org/abs/1803.06978>`_。

    参数：
        - **network** (Cell) - 目标模型。
        - **eps** (float) - 攻击产生的对抗性扰动占数据范围的比例。默认值：0.3。
        - **bounds** (tuple) - 数据的上下界，表示数据范围。以(数据最小值，数据最大值)的形式出现。默认值：(0.0, 1.0)。
        - **is_targeted** (bool) - 如果为True，则为目标攻击。如果为False，则为无目标攻击。默认值：False。
        - **prob** (float) - 对输入样本的转换概率。默认值：0.5。
        - **loss_fn** (Union[Loss, None]) - 用于优化的损失函数。如果为None，则输入网络已配备损失函数。默认值：None。

    .. py:method:: generate(inputs, labels)

        基于多样性输入迭代法生成对抗样本。

        参数：
            - **inputs** (Union[numpy.ndarray, tuple]) - 良性输入样本，用于创建对抗样本。
            - **labels** (Union[numpy.ndarray, tuple]) - 原始/目标标签。若每个输入有多个标签，将它包装在元组中。

        返回：
            - **numpy.ndarray** - 生成的对抗样本。

.. py:class:: mindarmour.adv_robustness.attacks.MomentumDiverseInputIterativeMethod(network, eps=0.3, bounds=(0.0, 1.0), is_targeted=False, norm_level='l1', prob=0.5, loss_fn=None)

    动量多样性输入迭代法（Momentum Diverse Input Iterative Method）攻击是一种动量迭代法，在每次迭代时对输入数据应用随机变换。对输入数据的这种转换可以提高对抗样本的可转移性。

    参考文献：`Xie, Cihang and Zhang, et al., "Improving Transferability of Adversarial Examples With Input Diversity," in CVPR, 2019 <https://arxiv.org/abs/1803.06978>`_。

    参数：
        - **network** (Cell) - 目标模型。
        - **eps** (float) - 攻击产生的对抗性扰动占数据范围的比例。默认值：0.3。
        - **bounds** (tuple) - 数据的上下界，表示数据范围。以(数据最小值，数据最大值)的形式出现。默认值：(0.0, 1.0)。
        - **is_targeted** (bool) - 如果为True，则为目标攻击。如果为False，则为无目标攻击。默认值：False。
        - **norm_level** (Union[int, str, numpy.inf]) - 范数类型。
          可取值：numpy.inf、1、2、'1'、'2'、'l1'、'l2'、'np.inf'、'inf'、'linf'。默认值：'l1'。
        - **prob** (float) - 对输入样本的转换概率。默认值：0.5。
        - **loss_fn** (Union[Loss, None]) - 用于优化的损失函数。如果为None，则输入网络已配备损失函数。默认值：None。

.. py:class:: mindarmour.adv_robustness.attacks.DeepFool(network, num_classes, model_type='classification', reserve_ratio=0.3, max_iters=50, overshoot=0.02, norm_level=2, bounds=None, sparse=True)

    DeepFool是一种无目标的迭代攻击，通过将良性样本移动到最近的分类边界并跨越边界来实现。

    参考文献：`DeepFool: a simple and accurate method to fool deep neural networks <https://arxiv.org/abs/1511.04599>`_。

    参数：
        - **network** (Cell) - 目标模型。
        - **num_classes** (int) - 模型输出的标签数，应大于零。
        - **model_type** (str) - 目标模型的类型。现在支持'classification'和'detection'。默认值：'classification'。
        - **reserve_ratio** (Union[int, float]) - 攻击后可检测到的对象百分比，仅当model_type='detection'时有效。保留比率应在(0, 1)的范围内。默认值：0.3。
        - **max_iters** (int) - 最大迭代次数，应大于零。默认值：50。
        - **overshoot** (float) - 过冲参数。默认值：0.02。
        - **norm_level** (Union[int, str, numpy.inf]) - 矢量范数类型。可取值：numpy.inf或2。默认值：2。
        - **bounds** (Union[tuple, list]) - 数据范围的上下界。以(数据最小值，数据最大值)的形式出现。默认值：None。
        - **sparse** (bool) - 如果为True，则输入标签为稀疏编码。如果为False，则输入标签为one-hot编码。默认值：True。

    .. py:method:: generate(inputs, labels)    

        根据输入样本和原始标签生成对抗样本。

        参数：
            - **inputs** (Union[numpy.ndarray, tuple]) - 输入样本。

              - 如果 `model_type` ='classification'，则输入的格式应为numpy.ndarray。输入的格式可以是(input1, input2, ...)。
              - 如果 `model_type` ='detection'，则只能是一个数组。

            - **labels** (Union[numpy.ndarray, tuple]) - 目标标签或ground-truth标签。

              - 如果 `model_type` ='classification'，标签的格式应为numpy.ndarray。
              - 如果 `model_type` ='detection'，标签的格式应为(gt_boxes, gt_labels)。

        返回：
            - **numpy.ndarray** - 对抗样本。

        异常：
            - **NotImplementedError** - `norm_level` 不在[2, numpy.inf, '2', 'inf']中。

.. py:class:: mindarmour.adv_robustness.attacks.CarliniWagnerL2Attack(network, num_classes, box_min=0.0, box_max=1.0, bin_search_steps=5, max_iterations=1000, confidence=0, learning_rate=5e-3, initial_const=1e-2, abort_early_check_ratio=5e-2, targeted=False, fast=True, abort_early=True, sparse=True)

    使用L2范数的Carlini & Wagner攻击通过分别利用两个损失生成对抗样本：“对抗损失”可使生成的示例实际上是对抗性的，“距离损失”可以控制对抗样本的质量。

    参考文献：`Nicholas Carlini, David Wagner: "Towards Evaluating the Robustness of Neural Networks" <https://arxiv.org/abs/1608.04644>`_。

    参数：
        - **network** (Cell) - 目标模型。
        - **num_classes** (int) - 模型输出的标签数，应大于零。
        - **box_min** (float) - 目标模型输入的下界。默认值：0。
        - **box_max** (float) - 目标模型输入的上界。默认值：1.0。
        - **bin_search_steps** (int) - 用于查找距离和置信度之间的最优trade-off常数的二分查找步数。默认值：5。
        - **max_iterations** (int) - 最大迭代次数，应大于零。默认值：1000。
        - **confidence** (float) - 对抗样本输出的置信度。默认值：0。
        - **learning_rate** (float) - 攻击算法的学习率。默认值：5e-3。
        - **initial_const** (float) - 用于平衡扰动范数和置信度差异的初始trade-off常数。默认值：1e-2。
        - **abort_early_check_ratio** (float) - 检查所有迭代中所有比率的损失进度。默认值：5e-2。
        - **targeted** (bool) - 如果为True，则为目标攻击。如果为False，则为无目标攻击。默认值：False。
        - **fast** (bool) - 如果为True，则返回第一个找到的对抗样本。如果为False，则返回扰动较小的对抗样本。默认值：True。
        - **abort_early** (bool) - 是否提前终止。

          - 如果为True，则当损失在一段时间内没有减少，Adam将被中止。
          - 如果为False，Adam将继续工作，直到到达最大迭代。默认值：True。

        - **sparse** (bool) - 如果为True，则输入标签为稀疏编码。如果为False，则输入标签为one-hot编码。默认值：True。

    .. py:method:: generate(inputs, labels)

        根据输入数据和目标标签生成对抗样本。

        参数：
            - **inputs** (numpy.ndarray) - 输入样本。
            - **labels** (numpy.ndarray) - 输入样本的真值标签或目标标签。

        返回：
            - **numpy.ndarray** - 生成的对抗样本。

.. py:class:: mindarmour.adv_robustness.attacks.JSMAAttack(network, num_classes, box_min=0.0, box_max=1.0, theta=1.0, max_iteration=1000, max_count=3, increase=True, sparse=True)

    基于Jacobian的显著图攻击（Jacobian-based Saliency Map Attack）是一种基于输入特征显著图的有目标的迭代攻击。它使用每个类标签相对于输入的每个组件的损失梯度。然后，使用显著图来选择产生最大误差的维度。

    参考文献：`The limitations of deep learning in adversarial settings <https://arxiv.org/abs/1511.07528>`_。

    参数：
        - **network** (Cell) - 目标模型。
        - **num_classes** (int) - 模型输出的标签数，应大于零。
        - **box_min** (float) - 目标模型输入的下界。默认值：0。
        - **box_max** (float) - 目标模型输入的上界。默认值：1.0。
        - **theta** (float) - 一个像素的变化率（相对于输入数据范围）。默认值：1.0。
        - **max_iteration** (int) - 迭代的最大轮次。默认值：1000。
        - **max_count** (int) - 每个像素的最大更改次数。默认值：3。
        - **increase** (bool) - 如果为True，则增加扰动。如果为False，则减少扰动。默认值：True。
        - **sparse** (bool) - 如果为True，则输入标签为稀疏编码。如果为False，则输入标签为one-hot编码。默认值：True。

    .. py:method:: generate(inputs, labels) 

        批量生成对抗样本。

        参数：
            - **inputs** (numpy.ndarray) - 输入样本。
            - **labels** (numpy.ndarray) - 目标标签。

        返回：
            - **numpy.ndarray** - 对抗样本。

.. py:class:: mindarmour.adv_robustness.attacks.LBFGS(network, eps=1e-5, bounds=(0.0, 1.0), is_targeted=True, nb_iter=150, search_iters=30, loss_fn=None, sparse=False)

    L-BFGS-B攻击使用有限内存BFGS优化算法来最小化输入与对抗样本之间的距离。

    参考文献：`Pedro Tabacof, Eduardo Valle. "Exploring the Space of Adversarial Images" <https://arxiv.org/abs/1510.05328>`_。

    参数：
        - **network** (Cell) - 被攻击模型的网络。
        - **eps** (float) - 攻击步长。默认值：1e-5。
        - **bounds** (tuple) - 数据的上下界。默认值：(0.0, 1.0)
        - **is_targeted** (bool) - 如果为True，则为目标攻击。如果为False，则为无目标攻击。默认值：True。
        - **nb_iter** (int) - lbfgs优化器的迭代次数，应大于零。默认值：150。
        - **search_iters** (int) - 步长的变更数，应大于零。默认值：30。
        - **loss_fn** (Functions) - 替代模型的损失函数。默认值：None。
        - **sparse** (bool) - 如果为True，则输入标签为稀疏编码。如果为False，则输入标签为one-hot编码。默认值：False。

    .. py:method:: generate(inputs, labels)    

        根据输入数据和目标标签生成对抗样本。

        参数：
            - **inputs** (numpy.ndarray) - 良性输入样本，用于创建对抗样本。
            - **labels** (numpy.ndarray) - 原始/目标标签。

        返回：
            - **numpy.ndarray** - 生成的对抗样本。

.. py:class:: mindarmour.adv_robustness.attacks.GeneticAttack(model, model_type='classification', targeted=True, reserve_ratio=0.3, sparse=True, pop_size=6, mutation_rate=0.005, per_bounds=0.15, max_steps=1000, step_size=0.20, temp=0.3, bounds=(0, 1.0), adaptive=False, c=0.1)

    遗传攻击（Genetic Attack）为基于遗传算法的黑盒攻击，属于差分进化算法。

    此攻击是由Moustafa Alzantot等人（2018）提出的。 

    参考文献： `Moustafa Alzantot, Yash Sharma, Supriyo Chakraborty, "GeneticAttack: Practical Black-box Attacks with Gradient-FreeOptimization" <https://arxiv.org/abs/1805.11090>`_。

    参数：
        - **model** (BlackModel) - 目标模型。
        - **model_type** (str) - 目标模型的类型。现在支持'classification'和'detection'。默认值：'classification'。
        - **targeted** (bool) - 如果为True，则为目标攻击。如果为False，则为无目标攻击。 `model_type` ='detection'仅支持无目标攻击，默认值：True。
        - **reserve_ratio** (Union[int, float]) - 攻击后可检测到的对象百分比，仅当 `model_type` ='detection'时有效。保留比率应在(0, 1)的范围内。默认值：0.3。
        - **pop_size** (int) - 粒子的数量，应大于零。默认值：6。
        - **mutation_rate** (Union[int, float]) - 突变的概率，应在（0,1）的范围内。默认值：0.005。
        - **per_bounds** (Union[int, float]) - 扰动允许的最大无穷范数距离。
        - **max_steps** (int) - 每个对抗样本的最大迭代轮次。默认值：1000。
        - **step_size** (Union[int, float]) - 攻击步长。默认值：0.2。
        - **temp** (Union[int, float]) - 用于选择的采样温度。默认值：0.3。温度越大，个体选择概率之间的差异就越大。
        - **bounds** (Union[tuple, list, None]) - 数据的上下界。以(数据最小值，数据最大值)的形式出现。默认值：(0, 1.0)。
        - **adaptive** (bool) - 为True，则打开突变参数的动态缩放。如果为false，则打开静态突变参数。默认值：False。
        - **sparse** (bool) - 如果为True，则输入标签为稀疏编码。如果为False，则输入标签为one-hot编码。默认值：True。
        - **c** (Union[int, float]) - 扰动损失的权重。默认值：0.1。

    .. py:method:: generate(inputs, labels)    

        根据输入数据和目标标签（或ground_truth标签）生成对抗样本。

        参数：
            - **inputs** (Union[numpy.ndarray, tuple]) - 输入样本。

              - 如果 `model_type` ='classification'，则输入的格式应为numpy.ndarray。输入的格式可以是(input1, input2, ...)。
              - 如果 `model_type` ='detection'，则只能是一个数组。

            - **labels** (Union[numpy.ndarray, tuple]) - 目标标签或ground-truth标签。

              - 如果 `model_type` ='classification'，标签的格式应为numpy.ndarray。
              - 如果 `model_type` ='detection'，标签的格式应为(gt_boxes, gt_labels)。

        返回：
            - **numpy.ndarray** - 每个攻击结果的布尔值。
            - **numpy.ndarray** - 生成的对抗样本。
            - **numpy.ndarray** - 每个样本的查询次数。

.. py:class:: mindarmour.adv_robustness.attacks.HopSkipJumpAttack(model, init_num_evals=100, max_num_evals=1000, stepsize_search='geometric_progression', num_iterations=20, gamma=1.0, constraint='l2', batch_size=32, clip_min=0.0, clip_max=1.0, sparse=True)

    Chen、Jordan和Wainwright提出的HopSkipJumpAttack是一种基于决策的攻击。此攻击需要访问目标模型的输出标签。

    参考文献：`Chen J, Michael I. Jordan, Martin J. Wainwright. HopSkipJumpAttack: A Query-Efficient Decision-Based Attack. 2019. arXiv:1904.02144 <https://arxiv.org/abs/1904.02144>`_。

    参数：
        - **model** (BlackModel) - 目标模型。
        - **init_num_evals** (int) - 梯度估计的初始评估数。默认值：100。
        - **max_num_evals** (int) - 梯度估计的最大评估数。默认值：1000。
        - **stepsize_search** (str) - 表示要如何搜索步长；

          - 可取值为'geometric_progression'或'grid_search'。默认值：'geometric_progression'。
        - **num_iterations** (int) - 迭代次数。默认值：20。
        - **gamma** (float) - 用于设置二进制搜索阈值theta。默认值：1.0。
          对于l2攻击，二进制搜索阈值 `theta` 为 :math:`gamma / d^{3/2}` 。对于linf攻击是 :math:`gamma/d^2` 。默认值：1.0。
        - **constraint** (str) - 要优化距离的范数。可取值为'l2'或'linf'。默认值：'l2'。
        - **batch_size** (int) - 批次大小。默认值：32。
        - **clip_min** (float, optional) - 最小图像组件值。默认值：0。
        - **clip_max** (float, optional) - 最大图像组件值。默认值：1。
        - **sparse** (bool) - 如果为True，则输入标签为稀疏编码。如果为False，则输入标签为one-hot编码。默认值：True。

    异常：
        - **ValueError** - `stepsize_search` 不在['geometric_progression','grid_search']中。
        - **ValueError** - `constraint` 不在['l2', 'linf']中

    .. py:method:: generate(inputs, labels)    

        在for循环中生成对抗图像。

        参数：
            - **inputs** (numpy.ndarray) - 原始图像。
            - **labels** (numpy.ndarray) - 目标标签。

        返回：
            - **numpy.ndarray** - 每个攻击结果的布尔值。
            - **numpy.ndarray** - 生成的对抗样本。
            - **numpy.ndarray** - 每个样本的查询次数。

    .. py:method:: set_target_images(target_images)

        设置目标图像进行目标攻击。

        参数：
            - **target_images** (numpy.ndarray) - 目标图像。

.. py:class:: mindarmour.adv_robustness.attacks.NES(model, scene, max_queries=10000, top_k=-1, num_class=10, batch_size=128, epsilon=0.3, samples_per_draw=128, momentum=0.9, learning_rate=1e-3, max_lr=5e-2, min_lr=5e-4, sigma=1e-3, plateau_length=20, plateau_drop=2.0, adv_thresh=0.25, zero_iters=10, starting_eps=1.0, starting_delta_eps=0.5, label_only_sigma=1e-3, conservative=2, sparse=True)

    该类是自然进化策略（Natural Evolutionary Strategies，NES）攻击法的实现。NES使用自然进化策略来估计梯度，以提高查询效率。NES包括三个设置：Query-Limited设置、Partial-Information置和Label-Only设置。

    - 在'query-limit'设置中，攻击对目标模型的查询数量有限，但可以访问所有类的概率。
    - 在'partial-info'设置中，攻击仅有权访问top-k类的概率。
    - 在'label-only'设置中，攻击只能访问按其预测概率排序的k个推断标签列表。 

    在Partial-Information设置和Label-Only设置中，NES会进行目标攻击，因此用户需要使用set_target_images方法来设置目标类的目标图像。


    参考文献：`Andrew Ilyas, Logan Engstrom, Anish Athalye, and Jessy Lin. Black-box adversarial attacks with limited queries and information. In ICML, July 2018 <https://arxiv.org/abs/1804.08598>`_。

    参数：
        - **model** (BlackModel) - 要攻击的目标模型。
        - **scene** (str) - 确定算法的场景，可选值为：'Label_Only'、'Partial_Info'、'Query_Limit'。
        - **max_queries** (int) - 生成对抗样本的最大查询编号。默认值：10000。
        - **top_k** (int) - 用于'Partial-Info'或'Label-Only'设置，表示攻击者可用的（Top-k）信息数量。对于Query-Limited设置，此输入应设置为-1。默认值：-1。
        - **num_class** (int) - 数据集中的类数。默认值：10。
        - **batch_size** (int) - 批次大小。默认值：128。
        - **epsilon** (float) - 攻击中允许的最大扰动。默认值：0.3。
        - **samples_per_draw** (int) - 对偶采样中绘制的样本数。默认值：128。
        - **momentum** (float) - 动量。默认值：0.9。
        - **learning_rate** (float) - 学习率。默认值：1e-3。
        - **max_lr** (float) - 最大学习率。默认值：5e-2。
        - **min_lr** (float) - 最小学习率。默认值：5e-4。
        - **sigma** (float) - 随机噪声的步长。默认值：1e-3。
        - **plateau_length** (int) - 退火算法中使用的平台长度。默认值：20。
        - **plateau_drop** (float) - 退火算法中使用的平台Drop。默认值：2.0。
        - **adv_thresh** (float) - 对抗阈值。默认值：0.25。
        - **zero_iters** (int) - 用于代理分数的点数。默认值：10。
        - **starting_eps** (float) - Label-Only设置中使用的启动epsilon。默认值：1.0。
        - **starting_delta_eps** (float) - Label-Only设置中使用的delta epsilon。默认值：0.5。
        - **label_only_sigma** (float) - Label-Only设置中使用的Sigma。默认值：1e-3。
        - **conservative** (int) - 用于epsilon衰变的守恒，如果没有收敛，它将增加。默认值：2。
        - **sparse** (bool) - 如果为True，则输入标签为稀疏编码。如果为False，则输入标签为one-hot编码。默认值：True。

    .. py:method:: generate(inputs, labels)    

        根据输入数据和目标标签生成对抗样本。

        参数：
            - **inputs** (numpy.ndarray) - 良性输入样本。
            - **labels** (numpy.ndarray) - 目标标签。

        返回：
            - **numpy.ndarray** - 每个攻击结果的布尔值。
            - **numpy.ndarray** - 生成的对抗样本。
            - **numpy.ndarray** - 每个样本的查询次数。

        异常：
            - **ValueError** - 在'Label-Only'或'Partial-Info'设置中 `top_k` 小于0。
            - **ValueError** - 在'Label-Only'或'Partial-Info'设置中target_imgs为None。
            - **ValueError** - `scene` 不在['Label_Only', 'Partial_Info', 'Query_Limit']中

    .. py:method:: set_target_images(target_images)

        在'Partial-Info'或'Label-Only'设置中设置目标攻击的目标样本。

        参数：
            - **target_images** (numpy.ndarray) - 目标攻击的目标样本。


.. py:class:: mindarmour.adv_robustness.attacks.PointWiseAttack(model, max_iter=1000, search_iter=10, is_targeted=False, init_attack=None, sparse=True)

    点式攻击（Pointwise Attack）确保使用最小数量的更改像素为每个原始样本生成对抗样本。那些更改的像素将使用二进制搜索，以确保对抗样本和原始样本之间的距离尽可能接近。

    参考文献：`L. Schott, J. Rauber, M. Bethge, W. Brendel: "Towards the first adversarially robust neural network model on MNIST", ICLR (2019) <https://arxiv.org/abs/1805.09190>`_。

    参数：
        - **model** (BlackModel) - 目标模型。
        - **max_iter** (int) - 生成对抗图像的最大迭代轮数。默认值：1000。
        - **search_iter** (int) - 二进制搜索的最大轮数。默认值：10。
        - **is_targeted** (bool) - 如果为True，则为目标攻击。如果为False，则为无目标攻击。默认值：False。
        - **init_attack** (Union[Attack, None]) - 用于查找起点的攻击。默认值：None。
        - **sparse** (bool) - 如果为True，则输入标签为稀疏编码。如果为False，则输入标签为one-hot编码。默认值：True。


    .. py:method:: generate(inputs, labels)    

        根据输入样本和目标标签生成对抗样本。

        参数：
            - **inputs** (numpy.ndarray) - 良性输入样本，用于创建对抗样本。
            - **labels** (numpy.ndarray) - 对于有目标的攻击，标签是对抗性的目标标签。对于无目标攻击，标签是ground-truth标签。

        返回：
            - **numpy.ndarray** - 每个攻击结果的布尔值。
            - **numpy.ndarray** - 生成的对抗样本。
            - **numpy.ndarray** - 每个样本的查询次数。

.. py:class:: mindarmour.adv_robustness.attacks.PSOAttack(model, model_type='classification', targeted=False, reserve_ratio=0.3, sparse=True, step_size=0.5, per_bounds=0.6, c1=2.0, c2=2.0, c=2.0, pop_size=6, t_max=1000, pm=0.5, bounds=None)

    PSO攻击表示基于粒子群优化（Particle Swarm Optimization）算法的黑盒攻击，属于进化算法。
    此攻击由Rayan Mosli等人（2019）提出。 

    参考文献：`Rayan Mosli, Matthew Wright, Bo Yuan, Yin Pan, "They Might NOT Be Giants: Crafting Black-Box Adversarial Examples with Fewer Queries Using Particle Swarm Optimization", arxiv: 1909.07490, 2019. <https://arxiv.org/abs/1909.07490>`_。

    参数：
        - **model** (BlackModel) - 目标模型。
        - **step_size** (Union[int, float]) - 攻击步长。默认值：0.5。
        - **per_bounds** (Union[int, float]) - 扰动的相对变化范围。默认值：0.6。
        - **c1** (Union[int, float]) - 权重系数。默认值：2。
        - **c2** (Union[int, float]) - 权重系数。默认值：2。
        - **c** (Union[int, float]) - 扰动损失的权重。默认值：2。
        - **pop_size** (int) - 粒子的数量，应大于零。默认值：6。
        - **t_max** (int) - 每个对抗样本的最大迭代轮数，应大于零。默认值：1000。
        - **pm** (Union[int, float]) - 突变的概率，应在（0,1）的范围内。默认值：0.5。
        - **bounds** (Union[list, tuple, None]) - 数据的上下界。以(数据最小值，数据最大值)的形式出现。默认值：None。
        - **targeted** (bool) - 如果为True，则为目标攻击。如果为False，则为无目标攻击。 `model_type` ='detection'仅支持无目标攻击，默认值：False。
        - **sparse** (bool) - 如果为True，则输入标签为稀疏编码。如果为False，则输入标签为one-hot编码。默认值：True。
        - **model_type** (str) - 目标模型的类型。现在支持'classification'和'detection'。默认值：'classification'。
        - **reserve_ratio** (Union[int, float]) - 攻击后可检测到的对象百分比，用于 `model_type` ='detection'模式。保留比率应在(0, 1)的范围内。默认值：0.3。

    .. py:method:: generate(inputs, labels)

        根据输入数据和目标标签（或ground_truth标签）生成对抗样本。

        参数：
            - **inputs** (Union[numpy.ndarray, tuple]) - 输入样本。

              - 如果 `model_type` ='classification'，则输入的格式应为numpy.ndarray。输入的格式可以是(input1, input2, ...)。
              - 如果 `model_type` ='detection'，则只能是一个数组。

            - **labels** (Union[numpy.ndarray, tuple]) - 目标标签或ground-truth标签。

              - 如果 `model_type` ='classification'，标签的格式应为numpy.ndarray。
              - 如果 `model_type` ='detection'，标签的格式应为(gt_boxes, gt_labels)。

        返回：
            - **numpy.ndarray** - 每个攻击结果的布尔值。
            - **numpy.ndarray** - 生成的对抗样本。
            - **numpy.ndarray** - 每个样本的查询次数。

.. py:class:: mindarmour.adv_robustness.attacks.SaltAndPepperNoiseAttack(model, bounds=(0.0, 1.0), max_iter=100, is_targeted=False, sparse=True)

    增加椒盐噪声的量以生成对抗样本。

    参数：
        - **model** (BlackModel) - 目标模型。
        - **bounds** (tuple) - 数据的上下界。以(数据最小值，数据最大值)的形式出现。默认值：(0.0, 1.0)。
        - **max_iter** (int) - 生成对抗样本的最大迭代。默认值：100。
        - **is_targeted** (bool) - 如果为True，则为目标攻击。如果为False，则为无目标攻击。默认值：False。
        - **sparse** (bool) - 如果为True，则输入标签为稀疏编码。如果为False，则输入标签为one-hot编码。默认值：True。

    .. py:method:: generate(inputs, labels)

        根据输入数据和目标标签生成对抗样本。

        参数：
            - **inputs** (numpy.ndarray) - 原始的、未受扰动的输入。
            - **labels** (numpy.ndarray) - 目标标签。

        返回：
            - **numpy.ndarray** - 每个攻击结果的布尔值。
            - **numpy.ndarray** - 生成的对抗样本。
            - **numpy.ndarray** - 每个样本的查询次数。
