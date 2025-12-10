from mindspore import Parameter, Tensor, mint, nn, ops
import numpy as np
from mindformers.experimental.infer.core.transformer import ParallelTransformer, ParallelTransformerLayer



def _patched_layer_construct(self, x, freqs_cis=None, mask=None, batch_valid_length=None, block_tables=None,
                             slot_mapping=None, prefix_keys_values=None, q_seq_lens=None, key_cache=None, value_cache=None):
    """Construct function of transformer layer."""
    """But patch to save kv cache"""
    # hidden_states: [B, S, H]
    # norm at the beginning of the transformer layer.
    norm_output = self.attention_norm(x)
    # attention.
    attention_output = self.attention(norm_output, batch_valid_length, block_tables, slot_mapping, freqs_cis,
                                        mask, prefix_keys_values=prefix_keys_values,
                                        q_seq_lens=q_seq_lens, key_cache=key_cache, value_cache=value_cache)

    # residual-connection.
    if self.apply_residual_connection_post_norm:
        residual = norm_output
    else:
        residual = x
    norm_input = ops.add(residual, attention_output)
    # layernorm post attention.
    norm_output = self.ffn_norm(norm_input)
    # MLP.
    mlp_output = self.feed_forward(norm_output)
    # residual-connection.
    if self.apply_residual_connection_post_norm:
        residual = norm_output
    else:
        residual = norm_input
    output = ops.add(residual, mlp_output)
    return output, value_cache

ParallelTransformerLayer.construct = _patched_layer_construct


def _patch_network_construct(self, tokens: Tensor, batch_valid_length=None, batch_index=None, zactivate_len=None,
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
    hidden_states = self.cast(self.tok_embeddings(tokens), self.compute_dtype)
    # h: [bs, seq/1, hidden_dim]
    
    # only construct first layer
    i = 0
    prefix_kv = prefix_keys_values[i] if prefix_keys_values is not None else None
    key_cache_i = key_cache[i] if key_cache is not None else None
    value_cache_i = value_cache[i] if value_cache is not None else None
    # hidden_states = self.layers[i](hidden_states, freqs_cis, mask, batch_valid_length=batch_valid_length,
    #                                 block_tables=block_tables, slot_mapping=slot_mapping,
    #                                 prefix_keys_values=prefix_kv, q_seq_lens=q_seq_lens,
    #                                 key_cache=key_cache_i, value_cache=value_cache_i)
    hidden_states, hidden_value_cache = self.layers[i](hidden_states, freqs_cis, mask, batch_valid_length=batch_valid_length,
                                    block_tables=block_tables, slot_mapping=slot_mapping,
                                    prefix_keys_values=prefix_kv, q_seq_lens=q_seq_lens,
                                    key_cache=key_cache_i, value_cache=value_cache_i)


    if self.post_norm:
        hidden_states = self.norm_out(hidden_states)
    return hidden_states, hidden_value_cache

ParallelTransformer.construct = _patch_network_construct


from mindformers.models.utils import jit
from research.qwen2_5.infer.qwen2_5 import ParallelQwenForCausalLM
import mindspore.common.dtype as mstype
# pylint: disable=W0613

def _patch_qwen_causal_lm_construct(self, input_ids, labels=None, input_position=None, position_ids=None, attention_mask=None,
                input_embeds=None, init_reset=None, batch_valid_length=None, batch_index=None, zactivate_len=None,
                block_tables=None, slot_mapping=None, prefix_keys_values=None, llm_boost_inputs=None,
                q_seq_lens=None, key_cache=None, value_cache=None):
    """
    Forward of qwen model.
    """
    output, output_value_cache = self.model(input_ids, batch_valid_length, batch_index, zactivate_len, block_tables,
                        slot_mapping, prefix_keys_values, position_ids=position_ids, attention_mask=attention_mask,
                        q_seq_lens=q_seq_lens, key_cache=key_cache, value_cache=value_cache)
    if self.return_hidden_states:
        return output, output_value_cache
    pre_gather = (not self.use_past or self.is_first_iteration) and batch_valid_length is not None
    if pre_gather:
        batch_valid_length = mint.cumsum(batch_valid_length, 0)
        output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 0)
    logits = self.lm_head(output)

    logits = self.cast(logits, mstype.float32)
    if self.predict_run_mode:
        return self.reshape(logits, (-1, logits.shape[-1])), output_value_cache
    input_mask = self.cast(self.not_equal(input_ids, self.pad_token_id), mstype.float32)
    return logits, input_ids, input_mask, output_value_cache

ParallelQwenForCausalLM.construct = _patch_qwen_causal_lm_construct


from vllm_mindspore.model_executor.models.mf_models.mf_model_base import MfModelBase
from typing import Iterable, Optional, Set, Tuple, Union
from vllm.sequence import IntermediateTensors
from vllm.logger import init_logger
from mindformers.tools.utils import is_pynative
try:
    # Need to apply dllm pd patch on vllm to use pd disagg related functions
    from vllm.attention.layer import maybe_save_kv_layer_to_connector, wait_for_kv_layer_from_connector
    from vllm.distributed.kv_transfer import is_v1_kv_transfer_group
    kv_transfer_supported = True
except:
    kv_transfer_supported = False
    
logger = init_logger(__name__)

# 根据实际情况修改kvc存储的路径
KVC_FILE_PATH_PREFIX = "/workspace/h00672358/attack_kvc/infer/tensor_data/test/value_cache_"

def save_tensors_to_file(tensor_list, file_name):
    arr_list = [tensor.numpy().view(np.uint16) for tensor in tensor_list]
    np.savez(file_name, *arr_list)

def _patch_mf_model_base_forward(self,
            input_ids: Tensor,
            positions: Tensor,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds: Optional[Tensor] = None,
            **kwargs) -> Union[Tensor, IntermediateTensors]:
    model_inputs, is_prefill = self.prepare_inputs(input_ids, positions)
    model_inputs = self.update_model_inputs(model_inputs, **kwargs)

    # enable_mb_split is True in lager EP enable micro-batch and per-dp-bs > 1
    enable_mb_split = self.is_enable_micro_batch_split(
        is_prefill, model_inputs["q_seq_lens"])

    if is_prefill:
        if self.enable_micro_batch:
            self.network.phase = "prefill" if not enable_mb_split else "prefill_micro_batch"
            if not self.set_flags or is_pynative() or enable_mb_split:
                self.network.add_flags_custom(is_first_iteration=True)
                self.network.add_flags_enable_micro_batch(
                    enable_micro_batch=enable_mb_split)
        else:
            self.network.phase = "prefill"
            if not self.set_flags or is_pynative():
                self.network.add_flags_custom(is_first_iteration=True)

        hidden_states, hidden_value_cache = self.network(**model_inputs)
        
        block_table = model_inputs["block_tables"]
        block_ids = block_table[block_table != 0].tolist()
        target_value_cache = hidden_value_cache[block_ids]
        
        # kvc存储的路径
        file_name = KVC_FILE_PATH_PREFIX + str(model_inputs["input_ids"][0]) + "-" + str(model_inputs["input_ids"][-1])
        save_tensors_to_file([target_value_cache], file_name)
        

        
        self.network.phase = "increment"
        if not self.set_flags or is_pynative():
            self.network.add_flags_custom(is_first_iteration=False)
            self.set_flags = True
        if kv_transfer_supported:
            if is_v1_kv_transfer_group():
                self.connector_send_kvcache()
    else:
        if kv_transfer_supported:
            if is_v1_kv_transfer_group() and self.is_prefill_task():
                self.connector_send_kvcache()

            if is_v1_kv_transfer_group() and self.is_decoder_task():
                self.connector_wait_for_kv_layer()
                logger.debug(f"connector_wait_for_kv_layer success")
        hidden_states, hidden_value_cache = self.network(**model_inputs)

    return hidden_states

MfModelBase.forward = _patch_mf_model_base_forward

