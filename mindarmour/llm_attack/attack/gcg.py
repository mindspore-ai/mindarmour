'''
a light and fast GCG
'''
import copy
import random
import gc
import logging
import time

from dataclasses import dataclass
from typing import List, Union
import mindspore as ms
import numpy as np

logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

INIT_CHARS = [
    ".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}",
    "@", "#", "$", "%", "&", "*",
    "w", "x", "y", "z",
]

@dataclass
class GCGConfig:
    '''
    The parameter of GCG

    Args:
        num_steps (int): The number of steps to run the attack. Default 250.
        optim_str_init (Union[str, List[str]]): The initial string to optimize.
        search_width (int): The number of candidates to consider at each step. Default 512.
        batch_size (int): The batch size to use when computing losses. Default 200.
            If you run oom, try to decrease this value.
        topk (int): The number of top candidates to consider when sampling from grad.
        n_replace (int): The number of tokens to replace at each step. Default 1.
        buffer_size (int): The size of the attack buffer. Default 0.
        use_mellowmax (bool): Whether to use mellowmax for the losses. Default False.
        mellowmax_alpha (float): The alpha parameter for mellowmax. Default 1.0.
        early_stop (bool): Whether to stop early if a perfect match is found. Default False.
            It is recommended to set this to True. Use some easy-achieved target.
        use_prefix_cache (bool): Whether to use the prefix cache. Default True.
        allow_non_ascii (bool): Whether to allow non-ascii tokens in the optimization. Default False.
        filter_ids (bool): Whether to filter out ids that don't decode to the same string. Default True.
            This parameter is used when the code report runtime error. It is still in development.
        add_space_before_target (bool): Whether to add a space before the target string. Default False.
        seed (int): The seed to use for the attack. Default 20.
        verbosity (str): The verbosity level for the logger. Default "INFO".
            If you want to see less information, you can set it to "WARNING".
    '''
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: int = 200
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = 20
    verbosity: str = "INFO"

@dataclass
class GCGResult:
    '''
    GCGResult
    '''
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]

class AttackBuffer:
    '''
    AttackBuffer which stores the best strings and their losses. It is implemented as a min heap.
    Now still has some issues with the buffer size, it may doesn't work well as expected.
    '''
    def __init__(self, size):
        self.buffer = []
        self.size = size

    def add(self, loss, optim_ids) -> None:
        '''
        add the loss and optim_ids to the buffer
        '''
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self):
        '''
        return the top ids of the heap
        '''
        return self.buffer[0][1]

    def get_lowest_loss(self):
        '''
        return the lowest loss of the heap
        '''
        return self.buffer[0][0]

    def get_highest_loss(self):
        '''
        return the highest loss of the heap
        '''
        return self.buffer[-1][0]

    def log_buffer(self, tokenizer):
        '''
        log the buffer
        '''
        message = "buffer:"
        for loss, ids in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | string: {optim_str}"
        logger.info(message)

def set_seed(seed):
    '''
    set seed
    '''
    ms.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def mellowmax(t, alpha=1.0, axis=-1):
    '''
    use mellowmax as the losses
    '''
    logsumexp_result = ms.ops.logsumexp(alpha * t, axis=axis)
    log_t_shape = ms.ops.log(ms.tensor(t.shape[-1], dtype=t.dtype))
    return 1.0 / alpha * (logsumexp_result - log_t_shape)

class AttackManager():
    '''
    attack functions
    '''
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = None if config.allow_non_ascii else self.get_nonascii_toks()
        self.prefix_cache = None

        self.stop_flag = False

        self.target_ids = None
        self.before_embeds = None
        self.after_embeds = None
        self.target_embeds = None

    def get_nonascii_toks(self):
        '''
        This function is used to get the non-ascii tokens in the tokenizer.
        '''
        def is_ascii(s):
            return s.isascii() and s.isprintable()

        nonascii_toks = []
        for i in range(self.tokenizer.vocab_size):
            if not is_ascii(self.tokenizer.decode([i])):
                nonascii_toks.append(i)

        if self.tokenizer.bos_token_id is not None:
            nonascii_toks.append(self.tokenizer.bos_token_id)
        if self.tokenizer.eos_token_id is not None:
            nonascii_toks.append(self.tokenizer.eos_token_id)
        if self.tokenizer.pad_token_id is not None:
            nonascii_toks.append(self.tokenizer.pad_token_id)
        if self.tokenizer.unk_token_id is not None:
            nonascii_toks.append(self.tokenizer.unk_token_id)

        return ms.tensor(nonascii_toks)

    def init_buffer(self):
        '''
        initialize the buffer.
        '''
        tokenizer = self.tokenizer
        config = self.config

        logger.info("Initializing attack buffer of size %d...", config.buffer_size)

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)

        if isinstance(config.optim_str_init, str):
            init_optim_ids = tokenizer(
                config.optim_str_init, add_special_tokens=False, return_tensors="ms"
                )["input_ids"]
            optim_ids_len = init_optim_ids.shape[1]

            if config.buffer_size > 1:
                init_indices = ms.ops.randint(
                    0, len(INIT_CHARS), ((config.buffer_size - 1)*optim_ids_len,)
                )
                init_str = [INIT_CHARS[i] for i in init_indices]
                init_str = [
                    " ".join(init_str[i:i+optim_ids_len])
                    for i in range(0, len(init_str), optim_ids_len)
                    ]
                init_buffer_ids = tokenizer(
                    init_str, add_special_tokens=False, return_tensors="ms"
                    )["input_ids"]
                init_buffer_ids = ms.ops.cat([init_buffer_ids, init_optim_ids], axis=0)
            else:
                init_buffer_ids = init_optim_ids
        else: # assume list
            if len(config.optim_str_init) != config.buffer_size:
                logger.warning(
                    "Using %d initializations but buffer size is set to %d",
                    len(config.optim_str_init),
                    config.buffer_size
                )
            try:
                init_buffer_ids = tokenizer(
                    config.optim_str_init, add_special_tokens=False, return_tensors="pt"
                    )["input_ids"].type(ms.int64)
            except ValueError:
                logger.error(
                    "Unable to create buffer. Ensure that all initializations "
                    "tokenize to the same length."
                )

        true_buffer_size = max(1, config.buffer_size)

        # Compute the loss on the initial buffer entries
        if self.prefix_cache:
            init_buffer_embeds = ms.ops.cat([
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.tile((true_buffer_size, 1, 1)),
                self.target_embeds.tile((true_buffer_size, 1, 1)),
            ], axis=1)
        else:
            init_buffer_embeds = ms.ops.cat([
                self.before_embeds.tile((true_buffer_size, 1, 1)),
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.tile((true_buffer_size, 1, 1)),
                self.target_embeds.tile((true_buffer_size, 1, 1)),
            ], axis=1)

        init_buffer_losses = self.get_loss(init_buffer_embeds)

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        buffer.log_buffer(tokenizer)
        logger.info("Initialized attack buffer.")

        return buffer

    def token_gradient(self, input_ids):
        '''
        calculate the gradient of the input_ids
        '''
        def net(one_hot):

            input_embeds = one_hot @ self.embedding_layer.weight

            if self.config.use_prefix_cache:
                input_embeds = ms.ops.cat(
                    [input_embeds, self.after_embeds, self.target_embeds], axis=1
                    )
                output = self.model(
                    inputs_embeds=input_embeds, past_key_values=self.prefix_cache
                    )
            else:
                input_embeds = ms.ops.cat(
                    [self.before_embeds, input_embeds, self.after_embeds, self.target_embeds],
                    axis=1
                    )
                output = self.model(inputs_embeds=input_embeds)

            logits = output.logits

            # Shift logits so token n-1 predicts token n
            shift = input_embeds.shape[1] - self.target_ids.shape[1]
            shift_logits = logits[..., shift-1:-1, :].contiguous()
            shift_labels = self.target_ids.type(ms.int32)

            if self.config.use_mellowmax:
                label_logits = ms.ops.gather_elements(
                    shift_logits, -1, shift_labels.unsqueeze(-1)
                    ).squeeze(-1)
                loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, axis=-1)
            else:
                loss = ms.nn.CrossEntropyLoss()(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1)
                    )
            return loss

        one_hot = ms.ops.one_hot(
            input_ids,
            self.embedding_layer.weight.shape[0]
            ).type(self.model.dtype)
        net(one_hot)
        grad_fn = ms.grad(net)
        grad = grad_fn(one_hot)
        return grad

    def sample_control(self, control_toks, grad):
        '''
        choose Coordinate
        '''
        n_optim_tokens = len(control_toks)

        if self.config.allow_non_ascii is False:
            grad[:, self.not_allowed_ids] = float("inf")

        top_indices = ms.ops.topk((-grad), self.config.topk, dim=1)[1]
        original_control_toks = control_toks.tile((self.config.search_width, 1))
        new_token_pos = ms.ops.argsort(
            ms.ops.rand(self.config.search_width, n_optim_tokens)
            )[..., :self.config.n_replace]

        new_token_val = ms.ops.gather_elements(
            top_indices[new_token_pos], 2,
            ms.ops.randint(
                0,
                self.config.topk,
                (self.config.search_width, self.config.n_replace, 1)
                )
        ).squeeze(2)
        new_token_val = new_token_val.type(ms.int64)

        new_control_toks = ms.ops.tensor_scatter_elements(
            input_x=original_control_toks,
            axis=1,
            indices=new_token_pos,
            updates=new_token_val
        )
        return new_control_toks

    def get_filtered_cands(self, control_cand):
        '''
        make all Coordinate same length
        '''
        cands = []
        decoded_str = self.tokenizer.batch_decode(control_cand)
        for i, decoded in enumerate(decoded_str):
            encoded_str = self.tokenizer(
                decoded,
                add_special_tokens=False,
                return_tensors="ms")["input_ids"][0]
            if len(encoded_str) == len(control_cand[i]): # ms没有equal  equal是对标eq的
                if ms.ops.equal(encoded_str, control_cand[i]).all():
                    cands.append(control_cand[i])


        if not cands:
            # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
            raise RuntimeError(
                "No token sequences are the same after decoding and re-encoding. "
                "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
            )
        return ms.ops.stack(cands)

    def get_loss(self, input_embeds):
        '''
        This function is used to get the logits of the input_ids.
        '''
        all_loss = []
        prefix_cache_batch = []


        for i in range(0, input_embeds.shape[0], self.config.batch_size):
            input_embeds_batch = input_embeds[i:i+self.config.batch_size]
            current_batch_size = input_embeds_batch.shape[0]

            if self.prefix_cache:
                if not prefix_cache_batch or current_batch_size != self.config.batch_size:
                    prefix_cache_batch = [
                        [
                            x.broadcast_to((current_batch_size, -1, -1, -1))
                            for x in self.prefix_cache[i]
                        ]
                        for i in range(len(self.prefix_cache))
                    ]
                outputs = self.model(
                    inputs_embeds=input_embeds_batch,
                    past_key_values=prefix_cache_batch
                    )

            else:
                outputs = self.model(inputs_embeds=input_embeds_batch)

            logits = outputs.logits

            tmp = input_embeds.shape[1] - self.target_ids.shape[1]
            shift_logits = logits[..., tmp-1:-1, :].contiguous()
            shift_labels = self.target_ids.tile((current_batch_size, 1)).type(ms.int32)

            if self.config.use_mellowmax:
                label_logits = ms.ops.gather_elements(
                    shift_logits,
                    -1,
                    shift_labels.unsqueeze(-1)
                    ).squeeze(-1)
                loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, axis=-1)
            else:
                loss = ms.nn.CrossEntropyLoss(reduction="none")(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1)
                    )

            loss = loss.view(current_batch_size, -1).mean(axis=-1)
            all_loss.append(loss)

            if self.config.early_stop:
                if ms.ops.any(
                        ms.ops.all(
                            ms.ops.argmax(shift_logits, dim=-1) == shift_labels, axis=-1
                        )
                    ).item():
                    self.stop_flag = True

            del outputs
            gc.collect()


        return ms.ops.cat(all_loss, axis=0)

    def run(self, messages, target):
        '''
        run the attack
        '''
        if self.config.seed is not None:
            set_seed(self.config.seed)

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)

        if not any(["{optim_str}" in d["content"] for d in messages]):
            messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

        template = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
            )
        # Remove the BOS token -- this will get added when tokenizing, if necessary
        if self.tokenizer.bos_token and template.startswith(self.tokenizer.bos_token):
            template = template.replace(self.tokenizer.bos_token, "")
        before_str, after_str = template.split("{optim_str}")
        target = " " + target if self.config.add_space_before_target else target

        # Tokenize everything that doesn't get optimized
        before_ids = self.tokenizer(
            [before_str], padding=False, return_tensors="ms"
            )["input_ids"].type(ms.int64)

        after_ids = self.tokenizer(
            [after_str], add_special_tokens=False, return_tensors="ms"
            )["input_ids"].type(ms.int64)

        target_ids = self.tokenizer(
            [target], add_special_tokens=False, return_tensors="ms"
            )["input_ids"].type(ms.int64)

        # Embed everything that doesn't get optimized
        before_embeds, after_embeds, target_embeds = [
            self.embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)
            ]

        # Compute the KV Cache for tokens that appear before the optimized tokens
        if self.config.use_prefix_cache:
            output = self.model(inputs_embeds=before_embeds, use_cache=True)
            self.prefix_cache = output.past_key_values

        self.target_ids = target_ids
        self.before_embeds = before_embeds
        self.after_embeds = after_embeds
        self.target_embeds = target_embeds

        # Initialize the attack buffer
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []

        for i in range(self.config.num_steps):
            start = time.time()

            grad = self.token_gradient(optim_ids)

            sample = self.sample_control(optim_ids.squeeze(0), grad.squeeze(0))

            sampled = self.get_filtered_cands(sample)

            new_search_width = sampled.shape[0]
            if self.prefix_cache:
                input_embeds = ms.ops.cat([
                    self.embedding_layer(sampled),
                    self.after_embeds.tile((new_search_width, 1, 1)),
                    self.target_embeds.tile((new_search_width, 1, 1)),
                ], axis=1)
            else:
                input_embeds = ms.ops.cat([
                    self.before_embeds.tile((new_search_width, 1, 1)),
                    self.embedding_layer(sampled),
                    self.after_embeds.tile((new_search_width, 1, 1)),
                    self.target_embeds.tile((new_search_width, 1, 1)),
                ], axis=1)
            print(f"{new_search_width} candidates after filtering")
            loss = self.get_loss(input_embeds)

            current_loss = loss.min().item()
            optim_ids = sampled[loss.argmin()].unsqueeze(0)

            # Update the buffer based on the loss
            losses.append(current_loss)
            if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = self.tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            buffer.log_buffer(self.tokenizer)

            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.")
                break

            end = time.time()
            logger.info(
                "Step %d: loss=%.4f | time=%.2f",
                i, current_loss, end - start
                )

        min_loss_index = losses.index(min(losses))

        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
        )

        return result

# A wrapper around the GCG `run` method that provides a simple API
def run_gcg(model, tokenizer, messages, target, config):
    '''
    Generates a single optimized string using GCG.
    By now some tokenizer may not work properly,
    try to use some other tokenizer or report an issue when meet some error.

    Args:
        model: The model to attack.
        tokenizer: The tokenizer to use.
        messages: A list of dictionaries containing the harmful behavior in the conversation
            or just a behavior string.
        target: The target string want to generate at first.
        config: The GCGConfig object to use.

    Returns:
        A GCGResult object containing the best loss, best string, losses, and opitimized strings.
    '''
    if config is None:
        config = GCGConfig()

    logger.setLevel(getattr(logging, config.verbosity))

    gcg = AttackManager(model, tokenizer, config)
    result = gcg.run(messages, target)
    return result
