'''
Greedy Coordinate Gradient attack functions
'''
import gc
import logging

import mindspore as ms
from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


class Model:
    '''
    store the model and tokenizer
    '''
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       trust_remote_code=True,
                                                       use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            ms_dtype=ms.float16,
            low_cpu_mem_usage=True,
            use_cache=False
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = 'left'

    def get_nonascii_toks(self):
        '''
        This function is used to get the non-ascii tokens in the tokenizer.
        '''
        def is_ascii(s):
            return s.isascii() and s.isprintable()

        ascii_toks = []
        for i in range(3, self.tokenizer.vocab_size):
            if not is_ascii(self.tokenizer.decode([i])):
                ascii_toks.append(i)

        if self.tokenizer.bos_token_id is not None:
            ascii_toks.append(self.tokenizer.bos_token_id)
        if self.tokenizer.eos_token_id is not None:
            ascii_toks.append(self.tokenizer.eos_token_id)
        if self.tokenizer.pad_token_id is not None:
            ascii_toks.append(self.tokenizer.pad_token_id)
        if self.tokenizer.unk_token_id is not None:
            ascii_toks.append(self.tokenizer.unk_token_id)

        return ms.tensor(ascii_toks)

    def get_embedding_matrix(self):
        '''
        This function is used to get the embedding matrix of the model.
        '''
        return self.model.model.embed_tokens.weight

    def get_embeddings(self, input_ids):
        '''
        This function is used to get the embeddings of the input_ids.
        '''
        return self.model.model.embed_tokens(input_ids)


class AttackManager(Model):
    '''
    attack functions
    '''
    def __init__(self, model_path, batch_size, topk):
        super().__init__(model_path)
        self.batch_size = batch_size
        self.topk = topk

    def generate(self, input_ids, assistant_role_slice, gen_config=None):
        '''
        This function is used to generate the output_ids of the model.
        By default, only the first 32 tokens are generated.
        '''
        if gen_config is None:
            gen_config = self.model.generation_config
            gen_config.max_new_tokens = 32

        if gen_config.max_new_tokens > 50:
            print('WARNING: max_new_tokens > 32 may\
                 cause testing to slow down.')
        input_ids = input_ids[:assistant_role_slice.stop].unsqueeze(0)
        attn_masks = ms.ops.ones_like(input_ids)
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attn_masks,
            generation_config=gen_config,
            pad_token_id=self.tokenizer.pad_token_id
        )[0]

        return output_ids[assistant_role_slice.stop:]

    def check_for_attack_success(
            self,
            input_ids,
            assistant_role_slice,
            test_prefixes,
            gen_config=None,
            log=False
    ):
        '''
        This function is used to check if the model has been jailbroken.
        '''
        gen_str = self.tokenizer.decode(self.generate(
            input_ids,
            assistant_role_slice,
            gen_config=gen_config)).strip()

        if log:
            logger = logging.getLogger(__name__)
            logger.debug(gen_str)
        jailbroken = not any([prefix in gen_str for prefix in test_prefixes])
        return jailbroken

    def token_gradients(
            self,
            input_ids,
            input_slice,
            target_slice,
            loss_slice
    ):
        '''
        This function is used to calculate the gradients
        of the logits to adversial suffix.
        '''
        def net(one_hot):
            embed_weights = self.get_embedding_matrix()
            one_hot = ms.ops.tensor_scatter_elements(
                input_x=one_hot,
                axis=1,
                indices=input_ids[input_slice].unsqueeze(1),
                updates=ms.ops.ones(one_hot.shape[0],
                                    1,
                                    dtype=embed_weights.dtype)
            )

            input_embeds = (one_hot @ embed_weights).unsqueeze(0)
            # now stitch it together with the rest of the embeddings
            embeds = self.get_embeddings(input_ids.unsqueeze(0))
            full_embeds = ms.ops.cat(
                [
                    embeds[:, :input_slice.start, :],
                    input_embeds,
                    embeds[:, input_slice.stop:, :]
                ],
                axis=1)
            logits = self.model(inputs_embeds=full_embeds).logits
            targets = input_ids[target_slice]
            targets = targets.type(ms.int32)
            loss = ms.nn.CrossEntropyLoss()(logits[0, loss_slice, :], targets)
            return loss

        embed_weights = self.get_embedding_matrix()
        one_hot = ms.ops.zeros(
            input_ids[input_slice].shape[0],
            embed_weights.shape[0],
            dtype=embed_weights.dtype
        )
        net(one_hot)
        grad_fn = ms.grad(net)
        grad = grad_fn(one_hot)
        grad = grad / grad.norm(dim=-1, keepdim=True)
        return grad

    def sample_control(self, control_toks, grad, not_allowed_tokens=None):
        '''
        choose Coordinate
        '''
        if not_allowed_tokens is not None:
            grad[:, not_allowed_tokens] = np.infty
        top_indices = ms.ops.topk((-grad), self.topk, dim=1)[1]
        original_control_toks = control_toks.tile((self.batch_size, 1))
        new_token_pos = ms.ops.arange(
            0,
            len(control_toks),
            len(control_toks) / self.batch_size,
            dtype=ms.int64
        )

        new_token_val = ms.ops.gather_elements(
            top_indices[new_token_pos], 1,
            ms.ops.randint(0, self.topk, (self.batch_size, 1))
        )
        new_token_val = new_token_val.type(ms.int64)
        new_control_toks = ms.ops.tensor_scatter_elements(
            input_x=original_control_toks,
            axis=1,
            indices=new_token_pos.unsqueeze(1),
            updates=new_token_val
        )
        return new_control_toks

    def get_filtered_cands(
            self,
            control_cand,
            filter_cand=True,
            curr_control=None
    ):
        '''
        make all Coordinate same length
        '''
        cands, error_count = [], 0
        for i in range(control_cand.shape[0]):
            decoded_str = self.tokenizer.decode(control_cand[i],
                                                skip_special_tokens=True)
            if filter_cand:
                if (decoded_str != curr_control and
                        len(self.tokenizer(
                            decoded_str,
                            add_special_tokens=False).input_ids)
                        ==
                        len(control_cand[i])):
                    cands.append(decoded_str)
                else:
                    error_count += 1
            else:
                cands.append(decoded_str)

        if filter_cand:
            cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        return cands

    def forward(self, *, input_ids, attention_mask, batch_size):
        '''
        This function is used to get the logits of the input_ids.
        '''
        logits = []
        for i in range(0, input_ids.shape[0], batch_size):

            batch_input_ids = input_ids[i:i+batch_size]
            if attention_mask is not None:
                batch_attention_mask = attention_mask[i:i+batch_size]
            else:
                batch_attention_mask = None

            logits.append(
                self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                ).logits
            )

            gc.collect()

        del batch_input_ids, batch_attention_mask

        return ms.ops.cat(logits, axis=0)

    def get_logits(
            self,
            input_ids,
            control_slice,
            test_controls=None,
            return_ids=False,
    ):
        '''
        This function is used to get the logits of the input_ids.
        '''
        if isinstance(test_controls[0], str):
            max_len = control_slice.stop - control_slice.start
            test_ids = [
                ms.tensor(
                    self.tokenizer(control,
                                   add_special_tokens=False
                                   ).input_ids[:max_len]
                )
                for control in test_controls
            ]
            pad_tok = 0
            length = [t.size for t in test_ids]
            assert all(leng == length[0] for leng in length) is True
        else:
            raise ValueError(f"test_controls must be a list of strings,\
            got {type(test_controls)}")
        test_ids = ms.ops.stack(test_ids)
        if test_ids[0].shape[0] != control_slice.stop - control_slice.start:
            raise ValueError((
                f"test_controls must have shape "
                f"(n, {control_slice.stop - control_slice.start}), "
                f"got {test_ids.shape}"
            ))

        locs = ms.ops.arange(control_slice.start, control_slice.stop).tile((
            test_ids.shape[0], 1))

        ids = ms.ops.scatter(
            input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1),
            1,
            locs,
            test_ids
        )
        if pad_tok >= 0:
            attn_mask = (ids != pad_tok).type(ids.dtype)
        else:
            attn_mask = None

        if return_ids:
            del locs, test_ids
            gc.collect()
            return (
                self.forward(
                    input_ids=ids,
                    attention_mask=attn_mask,
                    batch_size=self.batch_size
                ),
                ids
            )

        del locs, test_ids
        logits = self.forward(input_ids=ids,
                              attention_mask=attn_mask,
                              batch_size=self.batch_size)
        del ids
        gc.collect()
        return logits

    def target_loss(self, logits, ids, target_slice):
        '''
        cal loss
        '''
        crit = ms.nn.CrossEntropyLoss(reduction='none')
        loss_slice = slice(target_slice.start-1, target_slice.stop-1)
        loss = crit(logits[:, loss_slice, :].swapaxes(1, 2),
                    ids[:, target_slice])
        return loss.mean(axis=-1)
