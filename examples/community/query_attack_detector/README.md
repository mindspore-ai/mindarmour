# 基于相似度的对抗样本检测

## 描述

本算法主要针对黑盒查询攻击生成的对抗样本进行检测。由于攻击过程往往生成大量相似的查询图片，如果某个查询与之前的查询相似度大于一定阈值，则判定为攻击。

## 环境要求

Mindspore >= 1.9，
本算法为模型无关，不依赖于具体模型和框架。

## 引入相关包

```Python
from query_detector import QueryDetector
import numpy as np
```

## 评估示例

* 准备查询输入。随机初始化10张正常样本，在第1张正常样本上添加随机噪声生成20张相似样本（模拟黑盒查询攻击过程）。

```Python
np.random.seed(5)
benign_queries = np.random.randint(0, 255, [10, 224, 224, 3], np.uint8)
suspicious_queries = benign_queries[1] + np.random.rand(20, 224, 224, 3)
suspicious_queries = suspicious_queries.astype(np.uint8)
```

* 新建QueryDetector，调用detect函数返回对抗样本id。从第2张查询图片开始，后续的查询图片均被判定为攻击。

```Python
detector = QueryDetector(suspicious_queries[0])
adv_ids = detector.detect(suspicious_queries)
```

## 实验结果

[QueryDetector] Image: 0, max match: 0, attack_query: False<br>
[QueryDetector] Image: 1, max match: 50, attack_query: True<br>
[QueryDetector] Image: 2, max match: 50, attack_query: True<br>
[QueryDetector] Image: 3, max match: 50, attack_query: True<br>
[QueryDetector] Image: 4, max match: 50, attack_query: True<br>
[QueryDetector] Image: 5, max match: 50, attack_query: True<br>
[QueryDetector] Image: 6, max match: 50, attack_query: True<br>
[QueryDetector] Image: 7, max match: 50, attack_query: True<br>
[QueryDetector] Image: 8, max match: 50, attack_query: True<br>
[QueryDetector] Image: 9, max match: 50, attack_query: True<br>
[QueryDetector] Image: 10, max match: 50, attack_query: True<br>
[QueryDetector] Image: 11, max match: 50, attack_query: True<br>
[QueryDetector] Image: 12, max match: 50, attack_query: True<br>
[QueryDetector] Image: 13, max match: 50, attack_query: True<br>
[QueryDetector] Image: 14, max match: 50, attack_query: True<br>
[QueryDetector] Image: 15, max match: 50, attack_query: True<br>
[QueryDetector] Image: 16, max match: 50, attack_query: True<br>
[QueryDetector] Image: 17, max match: 50, attack_query: True<br>
[QueryDetector] Image: 18, max match: 50, attack_query: True<br>
[QueryDetector] Image: 19, max match: 50, attack_query: True<br>
[QueryDetector] positive num on test dataset: 19<br>
[QueryDetector] positive rate: 0.95<br>
检测出的对抗样本id：[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]<br>

