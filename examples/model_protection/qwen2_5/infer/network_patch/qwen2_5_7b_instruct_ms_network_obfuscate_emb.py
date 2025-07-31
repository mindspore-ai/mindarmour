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

import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Parameter, Tensor, mint, nn, ops
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer

from mindformers.experimental.infer.core.transformer import ParallelTransformer
from vllm_mindspore.model_executor.models.mf_models.qwen2_weight_processor import Qwen2WeightProcessor


_orig_init = ParallelTransformer.__init__

def _patched_init(self, config, *args, **kwargs):
    _orig_init(self, config, *args, **kwargs)
    self.p_inv = Parameter(Tensor(np.arange(config.hidden_size), mstype.int32),
                        name='p_inv', parallel_optimizer=False)
    self.emb_p_inv = Parameter(Tensor(np.arange(config.vocab_size), mstype.int32),
                           name='emb_p_inv', parallel_optimizer=False)
    self.permute = ops.Gather().set_device('CPU')
    self.recover = ops.Gather().set_device('CPU')

ParallelTransformer.__init__ = _patched_init


def _patched_construct(self, tokens: Tensor, batch_valid_length=None, batch_index=None, zactivate_len=None,
                      block_tables=None, slot_mapping=None, prefix_keys_values=None, position_ids=None, attention_mask=None,
                      q_seq_lens=None, key_cache=None, value_cache=None):
        """
        Forward of ParallelTransformer.

        Args:
            tokens: the tokenized inputs with datatype int32
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.
            block_tables (Tensor[int64]): Store mapping tables for each sequence.
            slot_mapping (Tensor[int32]): Store token cache physical slot index.
        Returns:
            output: Tensor, the output of ParallelTransformer
        """
        # preprocess
        mask = attention_mask
        if self.use_past:
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr.prefill()

                if prefix_keys_values is not None:
                    bs, seq_len = self.shape(tokens)
                    if mask is None:
                        mask = self.casual_mask(tokens)
                    prefix_length = prefix_keys_values[0].shape[2]
                    prefix_mask = Tensor(np.zeros((bs, 1, seq_len, prefix_length)), dtype=mask.dtype)
                    mask = self.concat((prefix_mask, mask))
            else:
                freqs_cis = self.freqs_mgr.chunk_with_decode(position_ids)
        else:
            bs, seq_len = self.shape(tokens)
            mask = self.casual_mask(tokens)
            freqs_cis = self.freqs_mgr(seq_len)
            if prefix_keys_values is not None:
                prefix_length = prefix_keys_values[0].shape[2]
                prefix_mask = Tensor(np.zeros((bs, 1, seq_len, prefix_length)), dtype=mask.dtype)
                mask = self.concat((prefix_mask, mask))

        # tokens: [bs, seq/1]
        tokens = self.permute(self.emb_p_inv, tokens, 0)
        hidden_states = self.cast(self.tok_embeddings(tokens), self.compute_dtype)
        # h: [bs, seq/1, hidden_dim]
        for i in range(self.num_layers):
            prefix_kv = prefix_keys_values[i] if prefix_keys_values is not None else None
            key_cache_i = key_cache[i] if key_cache is not None else None
            value_cache_i = value_cache[i] if value_cache is not None else None
            hidden_states = self.layers[i](hidden_states, freqs_cis, mask, batch_valid_length=batch_valid_length,
                                           block_tables=block_tables, slot_mapping=slot_mapping,
                                           prefix_keys_values=prefix_kv, q_seq_lens=q_seq_lens,
                                           key_cache=key_cache_i, value_cache=value_cache_i)
        hidden_states = self.cast(hidden_states, mstype.float16)
        hidden_states = self.recover(hidden_states, self.p_inv, axis=1)
        hidden_states = self.cast(hidden_states, self.compute_dtype)
        if self.post_norm:
            hidden_states = self.norm_out(hidden_states)
        return hidden_states

ParallelTransformer.construct = _patched_construct


_orig_infer_convert_outer_weight = Qwen2WeightProcessor.infer_convert_outer_weight

def _patched_infer_convert_outer_weight(self, src_hf_dir, hf_weight_map):
    """convert weight not in model"""
    _orig_infer_convert_outer_weight(self, src_hf_dir, hf_weight_map)
 
    p_inv_hf_name = "model.p_inv"
    p_inv_ms_name = self.convert_weight_name(p_inv_hf_name)
    np_data, _ = self.get_safetensor_from_file(p_inv_hf_name, src_hf_dir, hf_weight_map)
    self.parameter_dict[p_inv_ms_name] = ms.Parameter(ms.from_numpy(np_data).astype(ms.int32),
                                                      name=p_inv_ms_name,
                                                      requires_grad=False)
    emb_p_hf_name = "model.emb_p_inv"
    emb_p_ms_name = self.convert_weight_name(emb_p_hf_name)
    np_data, _ = self.get_safetensor_from_file(emb_p_hf_name, src_hf_dir, hf_weight_map)
    self.parameter_dict[emb_p_ms_name] = ms.Parameter(ms.from_numpy(np_data).astype(ms.int32),
                                                      name=emb_p_ms_name,
                                                      requires_grad=False)


Qwen2WeightProcessor.infer_convert_outer_weight = _patched_infer_convert_outer_weight


_orig_convert_weight_name = Qwen2WeightProcessor.convert_weight_name

def _patched_convert_weight_name(self, weight_name: str):
    """replace weight name"""
    weight_name = _orig_convert_weight_name(self, weight_name)
    weight_name = weight_name.replace('self_attn.kv_p', 'attention.kv_p')
    weight_name = weight_name.replace('self_attn.kv_p_inv', 'attention.kv_p_inv')
    weight_name = weight_name.replace('self_attn.p', 'attention.p')
    weight_name = weight_name.replace('self_attn.p_inv', 'attention.p_inv')
    return weight_name

Qwen2WeightProcessor.convert_weight_name = _patched_convert_weight_name