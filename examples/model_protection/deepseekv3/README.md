# DeepseekV3_671B模型混淆

提供DeepseekV3_671B模型的混淆和推理脚本。用户分别传入模型文件的路径和推理数据，返回混淆后的模型和推理结果。

## 环境准备

硬件环境：至少4台Atlas 800I A2（64G）服务器，已安装必要的驱动程序，并可连接至互联网

操作系统：openEuler或Ubuntu Linux

软件环境：

1. Python >= 3.9, < 3.12

2. CANN >= 8.0.0.beta1

3. 安装vllm-MindSpore可以参考[vllm-MindSpore安装页面](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_zh_cn/getting_started/installation/installation.md)


4. 安装MindArmour:

   - 从Gitee下载源码

     `git clone https://gitee.com/mindspore/mindarmour.git`

   - 编译并安装MindArmour

     `python setup.py install`

### 文件结构说明

```bash
deepseekv3
├── obfuscation
│   ├── deepseekv3_weight_obfuscation.py # 模型文件混淆脚本
│   └── config
│       ├── deepseekv3_obf_config_huggingface.yaml   # 适用于Huggingface模型权重的混淆策略配置文件1
│       └── deepseekv3_obf_config_ms.yaml # 适用于MindSpore Lab模型权重的混淆策略配置文件
└── infer
    ├── deepseekv3_obfuscate_inference.py # 混淆模型推理脚本
    └── network_patch
        ├── deepseekv3_ms_network_obfuscate.py   # 模型结构修改脚本
        └── rope_patch
            ├── ObfuscateFreqsMgr.py # RoPE编码混淆脚本
            └── __init__.py 
```

## 脚本说明及使用

### DeepseekV3_671B模型混淆

1. #### 模型权重文件混淆。

   脚本`deepseekv3_weight_obfuscate.py`对外提供deepseekv3_671b的混淆能力。

   **脚本使用方式：**

   ```python
   python deepseekv3_weight_obfuscate.py <src_model_path> <saved_model_path> <obf_config_path>
   ```

   **脚本输入参数：**

   - src_model_path：需要混淆的模型文件路径。
   - saved_model_path：混淆后的模型文件保存路径。
   - obf_config_path：模型混淆策略配置文件。


2. #### 混淆态模型文件推理。

   脚本`deepseekv3_obfuscate_inference.py`对外提供deepseekv3_671b模型混淆态推理能力。

   **脚本使用方式：**

   ```python
   python deepseekv3_obfuscate_inference.py --config <config_path>
   ```
   
   **脚本输入参数：**

   - config_path：模型推理配置文件        
   
   在`deepseekv3_obfuscate_inference.py`脚本中，通过一行代码实现deepseekv3_671b模型结构修改：

   ```python
   from network_patch import deepseekv3_ms_network_obfuscate
   ```