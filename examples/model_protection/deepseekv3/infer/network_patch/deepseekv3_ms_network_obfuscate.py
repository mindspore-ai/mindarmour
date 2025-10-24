import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import Parameter, Tensor, mint, nn, ops
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer

from research.deepseek3.deepseek3_model_infer import DeepseekV3Model, DeepseekV3Attention
from vllm_mindspore.model_executor.models.mf_models.deepseekv3_weight_processor import DeepseekV3WeightProcessor
from mindformers.modules.layers import FreqsMgr, SeqExtendMethod

from .rope_patch.obfuscate_rope import ObfuscateFreqsMgr

_orig_init = DeepseekV3Model.__init__

def _patched_init(self, config, *args, **kwargs):
    _orig_init(self, config, *args, **kwargs)
    self.token_p = Parameter(Tensor(np.arange(129280), mstype.int32),
                             name='p_inv', parallel_optimizer=False)
    self.emb_p_inv = Parameter(Tensor(np.arange(7168), mstype.int32),
                               name='emb_p_inv', parallel_optimizer=False)
    self.rope_p = Parameter(Tensor(np.arange(32), mstype.int32),
                            name='rope_p', parallel_optimizer=False)
    
    self.permute = ops.Gather().set_device('CPU')
    self.recover = ops.Gather().set_device('CPU')
    self.rope_permute = ops.Gather().set_device('CPU')
    self.freqs_mgr = ObfuscateFreqsMgr(head_dim=self.qk_rope_head_dim,
                                       seq_length=config.seq_length,
                                       max_position_embedding=config.max_position_embeddings,
                                       rotary_dtype=config.rotary_dtype,
                                       theta=config.theta,
                                       scaling_factor=config.scaling_factor,
                                       extend_method=config.extend_method,
                                       is_dynamic=config.is_dynamic,
                                       rope_p=self.rope_p,
                                       rope_permute=self.rope_permute)

DeepseekV3Model.__init__ = _patched_init


def _patched_construct(self, tokens: Tensor, h=None, batch_valid_length=None, batch_index=None, zactivate_len=None,
                       block_tables=None, slot_mapping=None, position_ids=None, q_seq_lens=None,
                       attention_mask=None, attn_padding_idx=None, attn_unpadding_idx=None, ffn_padding_idx=None,
                       ffn_unpadding_idx=None, key_cache=None):
    """
    Forward of deepseekv3 model.

    Args:
        tokens: the tokenized inputs with datatype int32cd ..
        batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
            prediction. Tensor of shape :math:`(batch_size,)`. Default None.
        batch_index(Tensor): The generated batch index when use continuous batching in LLM serving.
            Tensor of shape :math:`(batch_size,)`. Default None.
        zactivate_len(Tensor): The slice length of KVCache when use dynamic shape infer.
            Tensor of shape :math:`(seq_length,)`. Default None.
        block_tables(Tensor[int64]): Store mapping tables for each sequence.
        slot_mapping(Tensor[int32]): Store token cache physical slot index.

    Returns:
        output: Tensor, the output of deepseekv3 decoderlayer
    """
    # preprocess
    # print("token_p: ", self.token_p)
    # print("emb_p_inv: ", self.emb_p_inv)
    # print("rope_p: ", self.rope_p)
    tokens = self.permute(self.token_p, tokens, 0)
    mask = attention_mask
    if self.is_first_iteration:
        freqs_cis = self.freqs_mgr.prefill()
    else:
        freqs_cis = self.freqs_mgr.chunk_with_decode(position_ids)

    if not self.pre_process and self.pipeline_parallel:
        if h is None:
            raise ValueError("when pipeline stage is not 0, h can not be None.")
    else:
        h = self.cast(self.tok_embeddings(tokens), self.dtype)

    # for splitting dual batch
    split_input = None
    split_bvl = None
    split_bt = None
    split_sm = None
    split_qsl = None

    for i in range(self.num_layers):
        key_cache_i = key_cache[i] if key_cache is not None else None
        if (self.moe_config.first_k_dense_replace and i < self.moe_config.first_k_dense_replace) \
                or not (self.enable_micro_batch and self.is_first_iteration):
            h = self.layers[i](h, freqs_cis, mask, batch_valid_length=batch_valid_length,
                               block_tables=block_tables, slot_mapping=slot_mapping,
                               q_seq_lens=q_seq_lens, attn_padding_idx=attn_padding_idx,
                               attn_unpadding_idx=attn_unpadding_idx, ffn_padding_idx=ffn_padding_idx,
                               ffn_unpadding_idx=ffn_unpadding_idx, key_cache=key_cache_i)
        else:
            # split dual batch in prefilling
            if i == self.moe_config.first_k_dense_replace:
                split_input, split_bvl, split_bt, split_sm, split_qsl = self._split_micro_batch_input(h, \
                        batch_valid_length, block_tables, slot_mapping, q_seq_lens)
            split_input = self.layers[i](split_input, freqs_cis, mask, batch_valid_length=split_bvl,
                                         block_tables=split_bt, slot_mapping=split_sm,
                                         q_seq_lens=split_qsl, attn_padding_idx=attn_padding_idx,
                                         attn_unpadding_idx=attn_unpadding_idx, ffn_padding_idx=ffn_padding_idx,
                                         ffn_unpadding_idx=ffn_unpadding_idx, key_cache=key_cache_i)
            if i == self.num_layers - 1:
                h = mint.concat((split_input[0], split_input[1]), dim=0)
                
    h = self.cast(h, mstype.float16)
    h = self.recover(h, self.emb_p_inv, axis=1)
    h = self.cast(h, self.dtype)
    
    if self.post_process:
        h = self.norm_out(h)
    return h
    
    
DeepseekV3Model.construct = _patched_construct


_orig_infer_convert_outer_weight = DeepseekV3WeightProcessor.infer_convert_outer_weight

def _patched_infer_convert_outer_weight(self, src_hf_dir, hf_weight_map):
    """convert weight not in model"""
    _orig_infer_convert_outer_weight(self, src_hf_dir, hf_weight_map)
 
    token_p_hf_name = "model.token_p"
    token_p_ms_name = self.convert_weight_name(token_p_hf_name)
    np_data, _ = self.get_safetensor_from_file(token_p_hf_name, src_hf_dir, hf_weight_map)
    self.parameter_dict[token_p_ms_name] = ms.Parameter(ms.from_numpy(np_data).astype(ms.int32),
                                                        name=token_p_ms_name,
                                                        requires_grad=False)
    emb_p_hf_name = "model.emb_p_inv"
    emb_p_ms_name = self.convert_weight_name(emb_p_hf_name)
    np_data, _ = self.get_safetensor_from_file(emb_p_hf_name, src_hf_dir, hf_weight_map)
    self.parameter_dict[emb_p_ms_name] = ms.Parameter(ms.from_numpy(np_data).astype(ms.int32),
                                                      name=emb_p_ms_name,
                                                      requires_grad=False)
    
    rope_p_hf_name = "model.rope_p"
    rope_p_ms_name = self.convert_weight_name(rope_p_hf_name)
    np_data, _ = self.get_safetensor_from_file(rope_p_hf_name, src_hf_dir, hf_weight_map)
    self.parameter_dict[rope_p_ms_name] = ms.Parameter(ms.from_numpy(np_data).astype(ms.int32),
                                                       name=rope_p_ms_name,
                                                       requires_grad=False)


DeepseekV3WeightProcessor.infer_convert_outer_weight = _patched_infer_convert_outer_weight