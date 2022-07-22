mindarmour.utils
================

MindArmour的工具方法。

.. py:class:: mindarmour.utils.LogUtil

    日志记录模块。

    在长期运行的脚本中记录随时间推移的日志统计信息。

    异常：
        - **SyntaxError** - 创建此类异常。

    .. py:method:: add_handler(handler)

        添加日志模块支持的其他处理程序。

        参数：
            - **handler** (logging.Handler) - 日志模块支持的其他处理程序。

        异常：
            - **ValueError** - 输入handler不是logging.Handler的实例。

    .. py:method:: debug(tag, msg, *args)

        记录'[tag] msg % args'，严重性为'DEBUG'。

        参数：
            - **tag** (str) - Logger标记。
            - **msg** (str) - Logger消息。
            - **args** (Any) - 辅助值。

    .. py:method:: error(tag, msg, *args)

        记录'[tag] msg % args'，严重性为'ERROR'。

        参数：
            - **tag** (str) - Logger标记。
            - **msg** (str) - Logger消息。
            - **args** (Any) - 辅助值。

    .. py:method:: get_instance()

        获取类 `LogUtil` 的实例。

        返回：
            - **Object** - 类 `LogUtil` 的实例。

    .. py:method:: info(tag, msg, *args)

        记录'[tag] msg % args'，严重性为'INFO'。

        参数：
            - **tag** (str) - Logger标记。
            - **msg** (str) - Logger消息。
            - **args** (Any) - 辅助值。

    .. py:method:: set_level(level)

        设置此logger的日志级别，级别必须是整数或字符串。支持的级别为 'NOTSET'(integer: 0)、'ERROR'(integer: 1-40)、'WARNING'('WARN', integer: 1-30)、'INFO'(integer: 1-20)以及'DEBUG'(integer: 1-10)

        例如，如果logger.set_level('WARNING')或logger.set_level(21)，则在运行时将打印脚本中的logger.warn()和logger.error()，而logger.info()或logger.debug()将不会打印。

        参数：
            - **level** (Union[int, str]) - logger的级别。

    .. py:method:: warn(tag, msg, *args)

        记录'[tag] msg % args'，严重性为'WARNING'。

        参数：
            - **tag** (str) - Logger标记。
            - **msg** (str) - Logger消息。
            - **args** (Any) - 辅助值。

.. py:class:: mindarmour.utils.GradWrapWithLoss(network)

    构造一个网络来计算输入空间中损失函数的梯度，并由 `weight` 加权。

    参数：
        - **network** (Cell) - 要包装的目标网络。

    .. py:method:: construct(inputs, labels)

        使用标签和权重计算 `inputs` 的梯度。

        参数：
            - **inputs** (Tensor) - 网络的输入。
            - **labels** (Tensor) - 输入的标签。

        返回：
            - **Tensor** - 梯度矩阵。

.. py:class:: mindarmour.utils.GradWrap(network)

    构建一个网络，以计算输入空间中网络输出的梯度，并由 `weight` 加权，表示为雅可比矩阵。

    参数：
        - **network** (Cell) - 要包装的目标网络。

    .. py:method:: construct(*data)

        计算雅可比矩阵（jacobian matrix）。

        参数：
            - **data** (Tensor) - 数据由输入和权重组成。

              - inputs: 网络的输入。
              - weight: 每个梯度的权重，'weight'与'labels'的shape相同。

        返回：
            - **Tensor** - 雅可比矩阵。
