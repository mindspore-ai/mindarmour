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

def gen_colums_permuate_list_MLA(hidden_size_1, hidden_size_2, heads, rope_contained = False):
    pi = []
    pi_1_one = np.random.permutation(np.arange(0, hidden_size_1))
    if rope_contained:
        pi_2_one_base_sequence = np.arange(0, hidden_size_2)
        groups = [pi_2_one_base_sequence[i*2:(i+1)*2] for i in range(hidden_size_2 // 2)]
        np.random.shuffle(groups)
        pi_2_one = np.concatenate(groups)
    else:
        pi_2_one = np.random.permutation(np.arange(0, hidden_size_2))
    total_dims = hidden_size_1 + hidden_size_2
    for i in range(heads):
        pi += (pi_1_one + i * total_dims).tolist()
        pi += (pi_2_one + i * total_dims + hidden_size_1).tolist()
    return pi, pi_1_one.tolist(), pi_2_one.tolist()

def repeat_permute(permute_list, heads):
    pi = []
    permute_arr = np.array(permute_list)
    for i in range(heads):
        pi += (permute_arr + i * len(permute_list)).tolist()
    return pi
    
def get_rope_permute_list(rope_list: list):
    rope_permute_list = []
    for i in range(0, len(rope_list), 2):
        first_element = rope_list[i]
        original_group_idx = first_element // 2
        rope_permute_list.append(original_group_idx)
    return rope_permute_list

def test_deepseekv3_weight_obfuscate(src_path, saved_path, obf_config_path):
    with open(obf_config_path, 'r') as f:
        obf_config = yaml.safe_load(f)
    obf = ModelObfuscator(obf_config, obfuscate_scale=100)
    vocab_size = 129280
    hidden_size = 7168
    num_heads = 128
    layers = 61
    q_lora_rank = 1536
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    MLP_inter_dim = 18432
    moe_inter_dim = 2048
    shared_moe_inter_dim = 2048
    
    token_p, _, _ = gen_colums_permuate_list_MLA(vocab_size, 0, 1)
    token_p_inv = inv_permutation(token_p)
    emb_p, _, _ = gen_colums_permuate_list_MLA(hidden_size, 0 ,1)
    emb_p_inv = inv_permutation(emb_p)
    
    qa_p, _, _ = gen_colums_permuate_list_MLA(q_lora_rank, 0, 1)
    qa_p_inv = inv_permutation(qa_p)
    
    qb_p, qb_p_nope_one, qb_p_rope = gen_colums_permuate_list_MLA(qk_nope_head_dim, qk_rope_head_dim, num_heads, True)
    rope_p = get_rope_permute_list(qb_p_rope)
    # _, kva_p, kva_p_rope = gen_colums_permuate_list_MLA(kv_lora_rank, qk_rope_head_dim, 1)
    # kva_p_inv = inv_permutation(kva_p)
    
    _, kva_p_nope, _ = gen_colums_permuate_list_MLA(kv_lora_rank, 0, 1)
    ka_p_rope = qb_p_rope
    kva_p = kva_p_nope + (np.array(ka_p_rope) + len(kva_p_nope)).tolist()
    kva_p_nope_inv = inv_permutation(kva_p_nope)
    
    
    kb_p_nope_one = qb_p_nope_one
    vb_p, vb_p_one, _ = gen_colums_permuate_list_MLA(v_head_dim, 0, num_heads)
    kvb_p_one = kb_p_nope_one + (np.array(vb_p_one) + len(kb_p_nope_one)).tolist()
    kvb_p = repeat_permute(kvb_p_one, num_heads)
    vp_b_inv = inv_permutation(vb_p)
    
    # _, o_p, _  = gen_colums_permuate_list_MLA(hidden_size, 0, 1)
    o_p = emb_p
    o_p_inv = inv_permutation(o_p)
    
    mlp_up_p, _, _ = gen_colums_permuate_list_MLA(MLP_inter_dim, 0, 1)
    mlp_up_p_inv = inv_permutation(mlp_up_p)
    mlp_down_p = emb_p
    
    expert_up_p, _, _ = gen_colums_permuate_list_MLA(moe_inter_dim, 0, 1)
    expert_up_p_inv = inv_permutation(expert_up_p)
    expert_down_p = emb_p
    
    shared_expert_up_p, _, _ = gen_colums_permuate_list_MLA(shared_moe_inter_dim, 0, 1)
    shared_expert_up_p_inv = inv_permutation(shared_expert_up_p)
    shared_expert_down_p = emb_p
    
    

    obf_metadata = {"token_p": np.array(token_p), "token_p_inv": np.array(token_p_inv), "emb_p": np.array(emb_p), "emb_p_inv" : np.array(emb_p_inv), "qa_p" : np.array(qa_p), "qa_p_inv" : np.array(qa_p_inv), "qb_p" : np.array(qb_p), "kva_p" : np.array(kva_p), "kva_p_nope" :np.array(kva_p_nope), "kva_p_nope_inv" : np.array(kva_p_nope_inv), "kvb_p" :np.array(kvb_p), "vb_p_inv" :np.array(vp_b_inv), "o_p" :np.array(o_p), "op_inv" :np.array(o_p_inv), 
                    "mlp_up_p" : np.array(mlp_up_p), "mlp_up_p_inv" : np.array(mlp_up_p_inv), "mlp_down_p" : np.array(mlp_down_p), "expert_up_p" : np.array(expert_up_p), "expert_up_p_inv" :np.array(expert_up_p_inv), "expert_down_p" : np.array(expert_down_p), 
                    "shared_expert_up_p" : np.array(shared_expert_up_p), "shared_expert_up_p_inv" : np.array(shared_expert_up_p_inv), "shared_expert_down_p" : np.array(shared_expert_down_p), "rope_p" : np.array(rope_p)}

    obf.set_metadata(obf_metadata)
    metadata_mapping = {}
    metadata_mapping['model.token_p'] = "token_p"
    metadata_mapping['model.emb_p_inv'] = "emb_p_inv"
    metadata_mapping['model.rope_p'] = "rope_p"
    obf.set_save_metadata_mapping(metadata_mapping)
    obf.obfuscate_weight_files(src_path, saved_path=saved_path)
    
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python deepseekv3_weight_obfuscate.py <src_model_path> <saved_model_path> <obf_config_path>")
        sys.exit(1)
    src_path, saved_path, obf_config_path = sys.argv[1], sys.argv[2], sys.argv[3]
    test_deepseekv3_weight_obfuscate(src_path, saved_path, obf_config_path)