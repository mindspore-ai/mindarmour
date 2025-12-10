# 项目说明
本项目用于实现kv cache逆向攻击，在窃取kv cache数据后，将其恢复出原始prompt
原理说明:
1. 首先对目标模型，修改模型脚本，遍历全量词表获取全量kv cache数据（第一层kv cache）
2. 对于目标kv cache进行匹配，获取到token ids
3. 实现词表detokenizer，逆向恢复出原始prompt

# 文件说明
network_patch：修改的模型脚本，实现在推理过程返回和记录kv cache数据
get_vocab_kvc.py: 遍历全量词表token id，获取kv cache数据
kvc_inversion_attack：对于指定的kv cache数据，逆向获取原始prompt

# 使用说明
```python
def get_kvc_tensor_from_file():
    '''
    自定义实现，根据实际的kvc文件名和路径，将全量kvc读取到list中
    '''
    token_num_per_file = 6000
    block_size = 16
    block_num = token_num_per_file // block_size
    kvc_list = []
    for i in range(25):
        begin, end = token_num_per_file * i, token_num_per_file * (i + 1) - 1
        file_name = "/workspace/h00672358/attack_kvc/infer/tensor_data/vocab_target_value_cache_" + str(begin) + "-" + str(end) + ".npz"
        kvc_from_vocab = load_array_from_file(file_name)[0][:block_num] # 正常应该有6000/16 = 375个block，但文件实际上有379个block
        kvc_list.append(kvc_from_vocab)
    file_name = "/workspace/h00672358/attack_kvc/infer/tensor_data/vocab_target_value_cache_150000-151642.npz"
    kvc_from_vocab = load_array_from_file(file_name)[0]
    kvc_list.append(kvc_from_vocab)
    return kvc_list

kvc_list = get_kvc_tensor_from_file()
vocab_path = "/workspace/models/qwen2.5_7B_Instruct/vocab.json"
kvc_processor = KVCProcessor(kvc_list, vocab_path)
def progress_callback(current, total):
    progress = int(current / total * 100)
try:
    kvc_array = load_array_from_file("/workspace/h00672358/attack_kvc/infer/tensor_data/kvc.npz")[0]
    processed_result = self.kvc_processor.inversion(kvc_array, progress_callback)
except Exception as e:
    print("attack failed")
```