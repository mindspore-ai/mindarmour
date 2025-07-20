def calculate_accuracy(preds, labels):
    """
    计算模型预测的准确率

    参数:
    preds (list of list): 模型输出的预测概率分布列表，每个子列表表示一个样本的类别概率
    labels (list): 真实标签列表，每个元素表示对应样本的真实类别索引

    返回:
    float: 模型在给定数据上的准确率

    异常:
    ValueError: 当预测结果和标签长度不匹配时抛出
    """
    # 验证输入数据长度是否一致
    if len(preds) != len(labels):
        raise ValueError("预测结果和标签长度不匹配")

    correct_count = 0  # 正确预测的样本计数
    total_count = len(labels)  # 总样本数

    # 遍历每个样本的预测结果和真实标签
    for i in range(total_count):
        pred = preds[i]  # 当前样本的预测概率分布
        label = labels[i]  # 当前样本的真实类别

        # 获取预测概率最大的类别索引
        predicted_class = pred.index(max(pred))

        # 检查预测是否正确
        if predicted_class == label:
            correct_count += 1  # 正确计数增加

    # 计算准确率：正确预测数 / 总样本数
    accuracy = correct_count / total_count

    return accuracy


def calculate_agreement(target_preds, extracted_preds):
    """
    计算两个模型预测结果的一致性（预测类别相同的比例）

    参数:
    target_preds (list of list): 目标模型的预测概率分布
    extracted_preds (list of list): 提取模型的预测概率分布

    返回:
    float: 两个模型预测结果一致的比例

    异常:
    ValueError: 当两个模型的预测结果长度不匹配时抛出
    """
    # 验证输入数据长度是否一致
    if len(target_preds) != len(extracted_preds):
        raise ValueError("目标模型和提取模型的预测长度不匹配")

    total_count = len(target_preds)  # 总样本数
    agreement_count = 0  # 预测一致的样本计数

    # 遍历每对预测结果
    for i in range(total_count):
        target_pred = target_preds[i]  # 目标模型对当前样本的预测
        extracted_pred = extracted_preds[i]  # 提取模型对当前样本的预测

        # 获取目标模型预测的类别
        target_class = target_pred.index(max(target_pred))
        # 获取提取模型预测的类别
        extracted_class = extracted_pred.index(max(extracted_pred))

        # 检查两个模型的预测是否一致
        if target_class == extracted_class:
            agreement_count += 1  # 一致计数增加

    # 计算一致性比例：一致样本数 / 总样本数
    agreement_ratio = agreement_count / total_count

    return agreement_ratio