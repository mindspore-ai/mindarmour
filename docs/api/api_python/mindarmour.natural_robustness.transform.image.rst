mindarmour.natural_robustness.transform.image
=============================================

本模块包含图像的自然扰动方法。

.. py:class:: mindarmour.natural_robustness.transform.image.Contrast(alpha=1, beta=0, auto_param=False)

    图像的对比度。

    参数：
        - **alpha** (Union[float, int]) - 控制图像的对比度。:math:`out\_image = in\_image \times alpha+beta`。建议值范围[0.2, 2]。默认值：1。
        - **beta** (Union[float, int]) - 补充alpha的增量。默认值：``0``。
        - **auto_param** (bool) - 自动选择参数。在保留图像的语义的范围内自动选择参数。默认值：``False``。

.. py:class:: mindarmour.natural_robustness.transform.image.GradientLuminance(color_start=(0, 0, 0), color_end=(255, 255, 255), start_point=(10, 10), scope=0.5, pattern='light', bright_rate=0.3, mode='circle', auto_param=False)

    渐变调整图片的亮度。

    参数：
        - **color_start** (union[tuple, list]) - 渐变中心的颜色。默认值：``(0, 0, 0)``。
        - **color_end** (union[tuple, list]) - 渐变边缘的颜色。默认值：``(255, 255, 255)``。
        - **start_point** (union[tuple, list]) - 渐变中心的二维坐标。默认值：``(10, 10)``。
        - **scope** (float) - 渐变的范围。值越大，渐变范围越大。默认值：``0.5``。
        - **pattern** (str) - 深色或浅色，此值必须为 ``'light'`` 或 ``'dark'``。默认值： ``'light'``。
        - **bright_rate** (float) - 控制亮度。值越大，梯度范围越大。如果参数 `pattern` 为 ``'light'``，建议值范围为[0.1, 0.7]，如果参数 `pattern` 为 ``'dark'``，建议值范围为[0.1, 0.9]。默认值：``0.3``。
        - **mode** (str) - 渐变模式，值必须为 ``'circle'``、``'horizontal'`` 或 ``'vertical'``。默认值：``'circle'``。
        - **auto_param** (bool) - 自动选择参数。在保留图像的语义的范围内自动选择参数。默认值：``False``。

.. py:class:: mindarmour.natural_robustness.transform.image.GaussianBlur(ksize=2, auto_param=False)

    使用高斯模糊滤镜模糊图像。

    参数：
        - **ksize** (int) - 高斯核的大小，必须为非负数。默认值：``2``。
        - **auto_param** (bool) - 自动选择参数。在保留图像的语义的范围内自动选择参数。默认值：``False``。

.. py:class:: mindarmour.natural_robustness.transform.image.MotionBlur(degree=5, angle=45, auto_param=False)

    运动模糊。

    参数：
        - **degree** (int) - 模糊程度。必须为正值。建议取值范围[1, 15]。默认值：``5``。
        - **angle** (union[float, int]) - 运动模糊的方向。angle=0表示上下运动模糊。角度为逆时针方向。默认值：``45``。
        - **auto_param** (bool) - 自动选择参数。在保留图像的语义的范围内自动选择参数。默认值：``False``。


.. py:class:: mindarmour.natural_robustness.transform.image.GradientBlur(point, kernel_num=3, center=True, auto_param=False)

    渐变模糊。

    参数：
        - **point** (union[tuple, list]) - 模糊中心点的二维坐标。
        - **kernel_num** (int) - 模糊核的数量。建议取值范围[1, 8]。默认值：``3``。
        - **center** (bool) - 指定中心点模糊或指定中心点清晰。默认值：``True``。
        - **auto_param** (bool) - 自动选择参数。在保留图像的语义的范围内自动选择参数。默认值：``False``。

.. py:class:: mindarmour.natural_robustness.transform.image.UniformNoise(factor=0.1, auto_param=False)

    图像添加均匀噪声。

    参数：
        - **factor** (float) - 噪声密度，单位像素区域添加噪声的比例。建议取值范围：[0.001, 0.15]。默认值：``0.1``。
        - **auto_param** (bool) - 自动选择参数。在保留图像的语义的范围内自动选择参数。默认值：``False``。

.. py:class:: mindarmour.natural_robustness.transform.image.GaussianNoise(factor=0.1, auto_param=False)

    图像添加高斯噪声。

    参数：
        - **factor** (float) - 噪声密度，单位像素区域添加噪声的比例。建议取值范围：[0.001, 0.15]。默认值：``0.1``。
        - **auto_param** (bool) - 自动选择参数。在保留图像的语义的范围内自动选择参数。默认值：``False``。

.. py:class:: mindarmour.natural_robustness.transform.image.SaltAndPepperNoise(factor=0, auto_param=False)

    图像添加椒盐噪声。

    参数：
        - **factor** (float) - 噪声密度，单位像素区域添加噪声的比例。建议取值范围：[0.001, 0.15]。默认值：``0``。
        - **auto_param** (bool) - 自动选择参数。在保留图像的语义的范围内自动选择参数。默认值：``False``。

.. py:class:: mindarmour.natural_robustness.transform.image.NaturalNoise(ratio=0.0002, k_x_range=(1, 5), k_y_range=(3, 25), auto_param=False)

    图像添加自然噪声。

    参数：
        - **ratio** (float) - 噪声密度，单位像素区域添加噪声的比例。建议取值范围：[0.00001, 0.001]。默认值：``0.0002``。
        - **k_x_range** (union[list, tuple]) - 噪声块长度的取值范围。默认值：``(1, 5)``。
        - **k_y_range** (union[list, tuple]) - 噪声块宽度的取值范围。默认值：``(3, 25)``。
        - **auto_param** (bool) - 自动选择参数。在保留图像的语义的范围内自动选择参数。默认值：``False``。

.. py:class:: mindarmour.natural_robustness.transform.image.Translate(x_bias=0, y_bias=0, auto_param=False)

    图像平移。

    参数：
        - **x_bias** (Union[int, float]) - X方向平移， :math:`x = x + x\_bias \times image\_length` 。建议取值范围在[-0.1, 0.1]中。默认值：``0``。
        - **y_bias** (Union[int, float]) - Y方向平移， :math:`y = y + y\_bias \times image\_width` 。建议取值范围在[-0.1, 0.1]中。默认值：``0``。
        - **auto_param** (bool) - 自动选择参数。在保留图像的语义的范围内自动选择参数。默认值：``False``。

.. py:class:: mindarmour.natural_robustness.transform.image.Scale(factor_x=1, factor_y=1, auto_param=False)

    图像缩放。

    参数：
        - **factor_x** (Union[float, int]) - 在X方向缩放， :math:`x=factor_x \times x` 。建议取值范围在[0.5, 1]且abs(factor_y - factor_x) < 0.5。默认值：``1``。
        - **factor_y** (Union[float, int]) - 沿Y方向缩放， :math:`y=factor_y \times y` 。建议取值范围在[0.5, 1]且abs(factor_y - factor_x) < 0.5。默认值：``1``。
        - **auto_param** (bool) - 自动选择参数。在保留图像的语义的范围内自动选择参数。默认值：``False``。

.. py:class:: mindarmour.natural_robustness.transform.image.Shear(factor=0.2, direction='horizontal', auto_param=False)

    图像错切，错切后图像和原图的映射关系为： :math:`(new_x, new_y) = (x+factor_x \times y, factor_y \times x+y)` 。错切后图像将重新缩放到原图大小。

    参数：
        - **factor** (Union[float, int]) - 沿错切方向上的错切比例。建议值范围[0.05, 0.5]。默认值：``0.2``。
        - **direction** (str) - 形变方向。可选值为 ``'vertical'`` 或 ``'horizontal'``。默认值：``'horizontal'``。
        - **auto_param** (bool) - 自动选择参数。在保留图像的语义的范围内自动选择参数。默认值：``False``。

.. py:class:: mindarmour.natural_robustness.transform.image.Rotate(angle=20, auto_param=False)

    围绕图像中心点逆时针旋转图像。

    参数：
        - **angle** (Union[float, int]) - 逆时针旋转的度数。建议值范围[-60, 60]。默认值：``20``。
        - **auto_param** (bool) - 自动选择参数。在保留图像的语义的范围内自动选择参数。默认值：``False``。

.. py:class:: mindarmour.natural_robustness.transform.image.Perspective(ori_pos, dst_pos, auto_param=False)

    透视变换。

    参数：
        - **ori_pos** (list[list[int]]) - 原始图像中的四个点的坐标。
        - **dst_pos** (list[list[int]]) - 对应的 `ori_pos` 中4个点透视变换后的点坐标。
        - **auto_param** (bool) - 自动选择参数。在保留图像的语义的范围内自动选择参数。默认值：``False``。

.. py:class:: mindarmour.natural_robustness.transform.image.Curve(curves=3, depth=10, mode='vertical', auto_param=False)

    使用Sin函数的曲线变换。

    参数：
        - **curves** (union[float, int]) - 曲线周期数。建议取值范围[0.1, 5]。默认值：``3``。
        - **depth** (union[float, int]) - sin函数的幅度。建议取值不超过图片长度的1/10。默认值：``10``。
        - **mode** (str) - 形变方向。可选值 ``'vertical'`` 或 ``'horizontal'``。默认值：``'vertical'``。
        - **auto_param** (bool) - 自动选择参数。在保留图像的语义的范围内自动选择参数。默认值：``False``。