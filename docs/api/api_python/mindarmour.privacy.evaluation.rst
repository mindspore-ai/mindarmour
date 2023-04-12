mindarmour.privacy.evaluation
=============================

本模块提供了一些评估给定模型隐私泄露风险的方法。

.. py:class:: mindarmour.privacy.evaluation.MembershipInference(model, n_jobs=-1)

    成员推理是由Shokri、Stronati、Song和Shmatikov提出的一种用于推断用户隐私数据的灰盒攻击。它需要训练样本的loss或logits结果，隐私是指单个用户的一些敏感属性。

    有关详细信息，请参见： `教程 <https://mindspore.cn/mindarmour/docs/zh-CN/r2.0/test_model_security_membership_inference.html>`_。

    参考文献：`Reza Shokri, Marco Stronati, Congzheng Song, Vitaly Shmatikov. Membership Inference Attacks against Machine Learning Models. 2017. <https://arxiv.org/abs/1610.05820v2>`_。

    参数：
        - **model** (Model) - 目标模型。
        - **n_jobs** (int) - 并行运行的任务数量。-1表示使用所有处理器，否则n_jobs的值必须为正整数。

    异常：
        - **TypeError** - 模型的类型不是 :class:`mindspore.Model` 。
        - **TypeError** - `n_jobs` 的类型不是int。
        - **ValueError** - `n_jobs` 的值既不是-1，也不是正整数。

    .. py:method:: eval(dataset_train, dataset_test, metrics)

        评估目标模型的隐私泄露程度。
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
            - **attack_config** (Union[list, tuple]) - 攻击模型的参数设置。格式为

              .. code-block:: python

                  attack_config =
                      [{"method": "knn", "params": {"n_neighbors": [3, 5, 7]}},
                       {"method": "lr", "params": {"C": np.logspace(-4, 2, 10)}}]

              - 支持的方法有knn、lr、mlp和rf，每个方法的参数必须在可变参数的范围内。参数实现的提示可在下面找到：

                - `KNN <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_
                - `LR <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
                - `RF <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_
                - `MLP <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html>`_

        异常：
            - **KeyError** - `attack_config` 中的配置没有键{"method", "params"}。
            - **NameError** - `attack_config` 中的方法（不区分大小写）不在["lr", "knn", "rf", "mlp"]中。

.. py:class:: mindarmour.privacy.evaluation.ImageInversionAttack(network, input_shape, input_bound, loss_weights=(1, 0.2, 5))

    一种通过还原图像的深层表达来重建图像的攻击方法。

    参考文献：`Aravindh Mahendran, Andrea Vedaldi. Understanding Deep Image Representations by Inverting Them. 2014. <https://arxiv.org/pdf/1412.0035.pdf>`_。

    参数：
        - **network** (Cell) - 网络，用于推断图像的深层特征。
        - **input_shape** (tuple) - 单个网络输入的数据形状，应与给定网络一致。形状的格式应为(channel, image_width, image_height)。
        - **input_bound** (Union[tuple, list]) - 原始图像的像素范围，应该像[minimum_pixel, maximum_pixel]或(minimum_pixel, maximum_pixel)。
        - **loss_weights** (Union[list, tuple]) - InversionLoss中三个子损失的权重，可以调整以获得更好的结果。默认值：(1, 0.2, 5)。

    异常：
        - **TypeError** - 网络类型不是Cell。
        - **ValueError** - `input_shape` 的值有非正整数。
        - **ValueError** - `loss_weights` 的值有非正数。

    .. py:method:: evaluate(original_images, inversion_images, labels=None, new_network=None)

        通过三个指标评估还原图像的质量：原始图像和还原图像之间的平均L2距离和SSIM值，以及新模型对还原图像的推理结果在真实标签上的置信度平均值。

        参数：
            - **original_images** (numpy.ndarray) - 原始图像，其形状应为(img_num, channels, img_width, img_height)。
            - **inversion_images** (numpy.ndarray) - 还原图像，其形状应为(img_num, channels, img_width, img_height)。
            - **labels** (numpy.ndarray) - 原始图像的ground truth标签。默认值：None。
            - **new_network** (Cell) - 其结构包含self._network中所有网络，但加载了不同的模型文件。默认值：None。

        返回：
            - **float** - l2距离。
            - **float** - 平均ssim值。
            - **Union[float, None]** - 平均置信度。如果 `labels` 或 `new_network` 为None，则该值为None。

    .. py:method:: generate(target_features, iters=100)

        根据 `target_features` 重建图像。

        参数：
            - **target_features** (numpy.ndarray) - 原始图像的深度表示。 `target_features` 的第一个维度应该是img_num。需要注意的是，如果img_num等于1，则 `target_features` 的形状应该是(1, dim2, dim3, ...)。
            - **iters** (int) - 逆向攻击的迭代次数，应为正整数。默认值：100。

        返回：
            - **numpy.ndarray** - 重建图像，预计与原始图像相似。

        异常：
            - **TypeError** - target_features的类型不是numpy.ndarray。
            - **ValueError** - `iters` 的值都不是正整数.

