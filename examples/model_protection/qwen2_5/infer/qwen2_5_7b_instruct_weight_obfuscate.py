# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import yaml
import numpy as np
from mindarmour import ModelObfuscator

def inv_permutation(p):
    inv_p = [0]*len(p)
    for old_idx, new_idx in enumerate(p):
        inv_p[new_idx] = old_idx
    return inv_p

def gen_colums_permuate_list(hidden_size, num_heads=None, kv_num_heads=None, use_gqa=False):
    if num_heads is None:
        pi = np.random.permutation(np.arange(hidden_size)).tolist()
        return pi, None
    
    if hidden_size % num_heads != 0:
        return None, None
    
    head_dims = int(hidden_size / num_heads)
    pi = []
    kv_pi = []
    if use_gqa and kv_num_heads is not None:
        repeat_num = int(num_heads / kv_num_heads)
        for i in range(kv_num_heads):
            one_head_pi = np.random.permutation(np.arange(0, head_dims))
            kv_head_pi = (one_head_pi + i * head_dims).tolist()
            kv_pi += kv_head_pi
            for j in range(repeat_num):
                q_head_pi = (one_head_pi + (i * repeat_num + j) * head_dims).tolist()
                pi += q_head_pi
    else:
        for i in range(num_heads):
            pi += np.random.permutation(np.arange(i * head_dims, (i + 1) * head_dims)).tolist()
        kv_pi = None
    return pi, kv_pi

def test_qwen2_5_weight_obfuscate(src_path, saved_path, obf_config_path):
    with open(obf_config_path, 'r') as f:
        obf_config = yaml.safe_load(f)
    obf = ModelObfuscator(obf_config, obfuscate_scale=100)
    hidden_size = 3584
    num_heads = 28
    layers = 28
    num_key_value_heads = 4
    p, kv_p = gen_colums_permuate_list(hidden_size, num_heads, num_key_value_heads, True)
    p_inv = inv_permutation(p)
    kv_p_inv = inv_permutation(kv_p)
    alpha_k = np.random.randint(1, 100, size=[1, ]).astype(np.float16)
    alpha_q = 1 / alpha_k

    obf_metadata = {"attn_pi": np.array(p), "attn_kv_pi" : np.array(kv_p), "attn_kv_pi_inv" : np.array(kv_p_inv), "attn_pi_inv" : np.array(p_inv), "alpha_q" : alpha_q, "alpha_k" : alpha_k}
    obf.set_metadata(obf_metadata)
    metadata_mapping = {}
    metadata_mapping['model.p'] = "attn_pi"
    metadata_mapping['model.p_inv'] = "attn_pi_inv"
    # for i in range(layers):
    #     metadata_mapping[f"model.layers.{i}.self_attn.p_inv"] = "attn_pi_inv"
    #     metadata_mapping[f"model.layers.{i}.self_attn.p"] = "attn_pi"
    #     metadata_mapping[f"model.layers.{i}.self_attn.kv_p"] = "attn_kv_pi"
    #     metadata_mapping[f"model.layers.{i}.self_attn.kv_p_inv"] = "attn_kv_pi_inv"
    obf.set_save_metadata_mapping(metadata_mapping)
    obf.obfuscate_weight_files(src_path, saved_path=saved_path)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python qwen2_5_7b_instruct_weight_obfuscate.py <src_model_path> <saved_model_path> <obf_config_path>")
        sys.exit(1)
    src_path, saved_path, obf_config_path = sys.argv[1], sys.argv[2], sys.argv[3]
    test_qwen2_5_weight_obfuscate(src_path, saved_path, obf_config_path)
