# LLM_ATTAC

这里是基于mindnlp对论文["Universal and Transferable Adversarial Attacks on Aligned Language Models"](https://arxiv.org/abs/2307.15043)的复现，目前仅支持运行在GPU上的llama2系列模型。

## 实验

### 实验环境

除了MindArmour的依赖外，还需要

```markdown
python==3.9
cuda==11.6
mindspore==2.2.14
mindnlp==0.3.1
```

在运行7B模型时，请保证你的设备至少有28G以上的显存。你可以通过传入参数`batch_size`和`search_size`来防止OOM。其中，`batch_size`指的是每个`epoch`的搜索空间大小，`search_size`指的是该`batch`被拆分为`batch_size/search_size`次计算。

### 单模型单behavior

如果有目标的模型或者攻击prompt的话，请运行：

```shell
cd SingleModelSingleBehavior
python single.py
```

可以通过`python single.py --help`来获取更多传入参数的有关信息。

如果想在论文中的[数据集](examples\model_security\llm_attack\data\harmful_behaviors.csv)运行大规模实验，请将数据集下载到本地后放置到`/data`文件夹中，运行：

```shell
cd scripts
python exp_single.py
```

得到的结果会保存在`result`文件夹中。
