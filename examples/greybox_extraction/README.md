# Extract Model Attack 评估工具

![Model Extraction](https://via.placeholder.com/800x200.png?text=Model+Extraction+Attack+Visualization)

本项目提供了用于评估模型提取攻击效果的评估工具，包含两个核心函数：
1. **准确率计算**：评估目标模型在测试集上的分类性能
2. **一致性计算**：评估目标模型与提取模型预测结果的一致性

> **专为复现 Santiago Zanella-Béguelin 等人提出的自然语言模型灰盒提取攻击方法 (Extract Model Attack) 而设计**

## 核心功能

### 1. 准确率计算函数
`calculate_accuracy(preds, labels)`
- 计算模型预测的准确率
- 参数: 
  - `preds` (list of list): 模型输出的预测概率分布列表
  - `labels` (list): 真实标签列表
- 返回: 
  - `float`: 模型在给定数据上的准确率
- 异常: 当预测结果和标签长度不匹配时抛出ValueError

### 2. 一致性计算函数
`calculate_agreement(target_preds, extracted_preds)`
- 计算两个模型预测结果的一致性
- 参数: 
  - `target_preds` (list of list): 目标模型的预测概率分布
  - `extracted_preds` (list of list): 提取模型的预测概率分布
- 返回: 
  - `float`: 两个模型预测结果一致的比例
- 异常: 当两个模型的预测结果长度不匹配时抛出ValueError

## 使用示例

```python
# 定义目标模型的预测结果
target_preds = [
    [0.1, 0.2, 0.7],   # 预测类别2
    [0.4, 0.4, 0.2],   # 预测类别0
    [0.9, 0.05, 0.05]  # 预测类别0
]

# 定义提取模型的预测结果
extracted_preds = [
    [0.15, 0.1, 0.75],  # 预测类别2
    [0.35, 0.45, 0.2],   # 预测类别1
    [0.85, 0.1, 0.05]    # 预测类别0
]

# 定义真实标签
labels = [2, 0, 0]

# 计算评估指标
accuracy = calculate_accuracy(target_preds, labels)
agreement = calculate_agreement(target_preds, extracted_preds)

print(f"目标模型准确率: {accuracy:.2%}")
print(f"模型间一致性: {agreement:.2%}")

# 参考文献
## Zanella-Béguelin S, et al. "Grey-box Extraction of Natural Language Models". Microsoft Research, 2021. 