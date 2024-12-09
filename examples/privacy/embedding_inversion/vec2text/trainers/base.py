'''
base trainer
'''
import collections
import copy
import logging
import random
from typing import Callable, Dict, List, Tuple, Union

import evaluate
import nltk
import numpy as np
import scipy.stats
import tqdm
from mindnlp.engine import Trainer, EvalLoopOutput
import mindspore as ms
import mindspore.ops as ops

logger = logging.getLogger(__name__)

# pylint: disable=W0612
DEFAULT_INPUT_STRING = ("Twas brillig, and the slithy toves, Did gyre and gimble in the wabe,"
                        "All mimsy were the borogoves, And the mome raths outgrabe.")

# pylint: disable=W0613
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(axis=-1)


def sem(l: List[float]) -> float:
    result = scipy.stats.sem(np.array(l))
    if isinstance(result, np.ndarray):
        return result.mean().item()
    return result


def mean(l: Union[List[int], List[float]]) -> float:
    return sum(l) / len(l)


def count_overlapping_ngrams(s1: str, s2: str, n: int) -> int:
    ngrams_1 = nltk.ngrams(s1, n)
    ngrams_2 = nltk.ngrams(s2, n)
    ngram_counts_1 = collections.Counter(ngrams_1)
    ngram_counts_2 = collections.Counter(ngrams_2)
    total = 0
    for ngram, count in ngram_counts_1.items():
        total += min(count, ngram_counts_2[ngram])
    return total


class BaseTrainer(Trainer):

    '''BaseTrainer'''

    additional_metrics: List[Callable[..., Dict[str, float]]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.compute_metrics = self.compute_metrics_func
        self.metric_accuracy = evaluate.load("accuracy")
        self.metric_bleu = evaluate.load("sacrebleu")
        self.metric_rouge = evaluate.load("rouge")
        self.additional_metrics = []

        self.gen_kwargs = {
            "early_stopping": False,
            "num_beams": 1,
            "do_sample": False,
            "no_repeat_ngram_size": 0,
        }
    @property
    def pad_token_id(self) -> int:
        try:
            return self.model.encoder_decoder.config.pad_token_id
        except AttributeError:
            return self.tokenizer.pad_token_id

    @property
    def bos_token_id(self) -> int:
        try:
            return self.model.encoder_decoder.decoder_start_token_id
        except AttributeError:
            return self.tokenizer.bos_token_id

    def sanity_decode(self, input_string: str = None, max_length: int = 128):
        """Encodes and decodes a string as a sanity check."""
        if input_string is None:
            input_string = DEFAULT_INPUT_STRING
        self.model.eval()
        print("=" * 16, "Begin trainer sanity check", "=" * 16)
        print("\tInput to encode ->", input_string)
        inputs = self.embedder_tokenizer(
            input_string,
            return_tensors="ms",
            max_length=max_length,
            padding="max_length",
        )
        inputs = inputs
        gen_kwargs = copy.copy(self.gen_kwargs)
        gen_kwargs["min_length"] = 1
        gen_kwargs["max_length"] = max_length
        print("max_length:", gen_kwargs["max_length"])
        regenerated = self.generate(
            inputs={
                "embedder_input_ids": inputs["input_ids"],
                "embedder_attention_mask": inputs["attention_mask"],
            },
            generation_kwargs=gen_kwargs,
        )
        print("\tDecoded output shape -> ", regenerated.shape)
        output_string = self.tokenizer.decode(
            regenerated.flatten(), skip_special_tokens=True
        )
        print("\tDecoded output ->", output_string)
        print("=" * 16, "End trainer sanity check", "=" * 16)

    def _log_preds_table(self, table_key: str, decoded_preds: List[str], decoded_labels: List[str]):
        '''
        _log_preds_table
        '''
        if not self.args.use_wandb:
            return

        if not self.args.local_rank <= 0:
            return

        num_rows = 50
        idxs = random.choices(
            range(len(decoded_preds)), k=min(len(decoded_preds), num_rows)
        )

        data = []
        for idx in idxs:
            data.append([decoded_labels[idx], decoded_preds[idx]])


    def _get_decoded_sequences(self, dataset, n: int) -> Tuple[List[ms.Tensor], List[ms.Tensor]]:
        """Iterates through eval dataset and does decoding.

        TODO: do this better. We shouldn't need to iterate through eval set twice
        but I don't want to copy 1000 lines of code to change their eval loop...

        Probably want custom eval eventually. Also this depends on eval data being
        in the same order which is annoying.
        """
        assert not self.model.training

        gen_kwargs = copy.copy(self.gen_kwargs)
        all_preds = []
        all_labels = []
        for _, inputs in enumerate(tqdm.tqdm(dataset, desc="generating from val", leave=False)):
            # https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation#transformers.GenerationMixin.generate
            # inputs_cuda = {k: v.to(self.args.device) for k, v in inputs.items()}
            max_length = self.model.config.max_seq_length
            gen_kwargs["max_length"] = max_length
            self.model.set_train(False)

            inputs_one_col_dict = {
                "input_ids": inputs[0],
                "attention_mask": inputs[1],
                "labels": inputs[2],
                "length": inputs[3],
                "embedder_input_ids": inputs[4],
                "embedder_attention_mask": inputs[5]


            }
            generated_text = self.generate(inputs=inputs_one_col_dict, generation_kwargs=gen_kwargs)
            self.model.set_train(True)
            if generated_text.shape[1] < max_length:
                # Pad generated text to max length
                pad_tokens = (
                    ops.ones(
                        (generated_text.shape[0], max_length - generated_text.shape[1]),
                        dtype=ms.int64
                    )
                    * self.pad_token_id
                )
                generated_text = ops.cat((generated_text, pad_tokens), axis=1)

            # true_input_ids = inputs["input_ids"]
            true_input_ids = inputs[0]
            if true_input_ids.shape[1] < max_length:
                # Pad true text to max length
                # Pad generated text to max length
                pad_tokens = (
                    ops.ones(
                        (true_input_ids.shape[0], max_length - true_input_ids.shape[1]),
                        dtype=ms.int64
                    )
                    * self.pad_token_id
                )
                true_input_ids = ops.cat((true_input_ids, pad_tokens), axis=1)

            all_preds.extend(generated_text.asnumpy().tolist())
            all_labels.extend(true_input_ids.asnumpy().tolist())
            if len(all_preds) >= n:
                break
        return all_preds, all_labels

    def _compute_data_metrics(self, inputs: Dict[str, ms.Tensor]) -> Dict[str, float]:
        '''compute_data_metrics'''
        inputs_pad_tokens = (
            (inputs["input_ids"] == self.tokenizer.pad_token_id)
            .sum(axis=1)
            .float()
            .mean()
            .item()
        )
        embedder_inputs_pad_tokens = (
            (inputs["embedder_input_ids"] == self.embedder_tokenizer.pad_token_id)
            .sum(axis=1)
            .float()
            .mean()
            .item()
        )

        inputs_non_pad_tokens = inputs["input_ids"].shape[1] - inputs_pad_tokens
        embedder_inputs_non_pad_tokens = (
            inputs["input_ids"].shape[1] - embedder_inputs_pad_tokens
        )

        return {
            "encoder_decoder_inputs_pad_tokens": inputs_pad_tokens,
            "encoder_decoder_inputs_non_pad_tokens": inputs_non_pad_tokens,
            "embedder_inputs_pad_tokens": embedder_inputs_pad_tokens,
            "embedder_inputs_non_pad_tokens": embedder_inputs_non_pad_tokens,
        }

    def compute_metrics_func(self, eval_preds):
        '''
        compute_metrics_func
        '''
        preds = eval_preds.predictions
        labels = eval_preds.label_ids

        assert labels, "got empty labels for eval"
        assert (
            ms.tensor(preds).shape == ms.tensor(labels).shape
        ), f"preds.shape {preds.shape} / labels.shape {labels.shape}"

        # preds have the same shape as the labels.
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        accuracy_result = self.metric_accuracy.compute(
            predictions=preds, references=labels
        )

        return {**accuracy_result}

    def _text_comparison_metrics(self, predictions_ids, predictions_str, references_ids, references_str):
        '''text_comparison_metrics'''
        assert len(predictions_ids) == len(references_ids)
        assert len(predictions_ids) == len(predictions_str)
        assert len(predictions_str) == len(references_str)
        num_preds = len(predictions_ids)
        if not num_preds:
            return {}



        # Compute token, precision, recall, and ngram-level metrics.
        precision_sum = 0.0
        recall_sum = 0.0
        num_overlapping_words = []
        num_overlapping_bigrams = []
        num_overlapping_trigrams = []
        num_true_words = []
        num_pred_words = []
        f1s = []
        for i in range(num_preds):
            true_words = nltk.tokenize.word_tokenize(references_str[i])
            pred_words = nltk.tokenize.word_tokenize(predictions_str[i])
            num_true_words.append(len(true_words))
            num_pred_words.append(len(pred_words))

            true_words_set = set(true_words)
            pred_words_set = set(pred_words)
            tp = len(true_words_set & pred_words_set)
            fp = len(true_words_set) - len(true_words_set & pred_words_set)
            fn = len(pred_words_set) - len(true_words_set & pred_words_set)

            precision = (tp) / (tp + fp + 1e-20)
            recall = (tp) / (tp + fn + 1e-20)

            try:
                f1 = (2 * precision * recall) / (precision + recall + 1e-20)
            except ZeroDivisionError:
                f1 = 0.0
            f1s.append(f1)

            precision_sum += precision
            recall_sum += recall

            ############################################################
            num_overlapping_words.append(
                count_overlapping_ngrams(true_words, pred_words, 1)
            )
            num_overlapping_bigrams.append(
                count_overlapping_ngrams(true_words, pred_words, 2)
            )
            num_overlapping_trigrams.append(
                count_overlapping_ngrams(true_words, pred_words, 3)
            )

        set_token_metrics = {
            "token_set_precision": (precision_sum / num_preds),
            "token_set_recall": (recall_sum / num_preds),
            "token_set_f1": mean(f1s),
            # "token_set_f1_sem": sem(f1s),
            # "n_ngrams_match_1": mean(num_overlapping_words),
            # "n_ngrams_match_2": mean(num_overlapping_bigrams),
            # "n_ngrams_match_3": mean(num_overlapping_trigrams),
            # "num_true_words": mean(num_true_words),
            # "num_pred_words": mean(num_pred_words),
        }
        ############################################################
        bleu_results = np.array(
            [
                self.metric_bleu.compute(predictions=[p], references=[r])["score"]
                for p, r in zip(predictions_str, references_str)
            ]
        )
        #rouge_result = self.metric_rouge.compute(
            #predictions=predictions_str, references=references_str
        #)
        self.bleu_results = (
            bleu_results.tolist()
        )  # store bleu results in case we want to use them later for t-tests
        # bertscore_result = self.metric_bertscore.compute(
        #     predictions=predictions_str, references=references_str, lang="en"
        # )
        exact_matches = np.array(predictions_str) == np.array(references_str)
        gen_metrics = {
            "bleu_score": bleu_results.mean(),
            # "bleu_score_sem": sem(bleu_results),
            # "rouge_score": rouge_result[
            #     "rouge1"
            # ],  # ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
            # "bert_score": statistics.fmean(bertscore_result["f1"]),
            "exact_match": mean(exact_matches),
            # "exact_match_sem": sem(exact_matches),
        }

        all_metrics = {**set_token_metrics, **gen_metrics}
        for metric in self.additional_metrics:
            all_metrics.update(metric(references_str, predictions_str))

        return all_metrics
    # pylint: disable=R0915
    # pylint: disable=W0212
    def eval_generation_metrics(self, dataset) -> Dict[str, float]:
        '''
        eval_generation_metrics
        '''
        # Get decoded text. Note that this is different than `preds`, which
        # is used to compute the loss.
        preds_sample_list, preds_sample_labels_list = self._get_decoded_sequences(
            dataset, n=10000
        )
        decoded_preds = self.tokenizer.batch_decode(
            preds_sample_list, skip_special_tokens=True
        )
        decoded_labels = self.tokenizer.batch_decode(
            preds_sample_labels_list, skip_special_tokens=True
        )
        bleu_result = self._text_comparison_metrics(
            predictions_ids=preds_sample_list,
            predictions_str=decoded_preds,
            references_ids=preds_sample_labels_list,
            references_str=decoded_labels,
        )
        #pylint: disable=W0613
        self._log_preds_table(
            table_key="val_text_preds",
            decoded_preds=decoded_preds,
            decoded_labels=decoded_labels,
        )

        if not decoded_preds:
            return {}
        print("[pred]", decoded_preds[3])
        print("[true]", decoded_labels[3])
        print("\n\n")
        print("[pred]", decoded_preds[1])
        print("[true]", decoded_labels[1])
        print("\n\n")
        print("[pred]", decoded_preds[2])
        print("[true]", decoded_labels[2])
        print("\n\n")

        # Compute sims of eval data using embedder.
        preds_sample = ms.tensor(preds_sample_list)[:128]
        preds_sample_labels = ms.tensor(
            preds_sample_labels_list
        )[:128]

        # Log num tokens.
        num_tokens_metrics = {
            "pred_num_tokens": (
                (preds_sample != self.pad_token_id)
                & (preds_sample != self.bos_token_id)).sum(1).float().mean().item(),
            "true_num_tokens": (
                (preds_sample_labels != self.pad_token_id)
                & (preds_sample_labels != self.bos_token_id)
            ).sum(1).float().mean().item(),}

        eos_token_id = self.embedder_tokenizer.eos_token_id
        if eos_token_id is not None:
            eos_tokens = (
                ops.ones(
                    (len(preds_sample), 1),
                    dtype=ms.int64
                )
                * eos_token_id
            )
            preds_sample = ops.cat((preds_sample[:, 1:], eos_tokens), axis=1)

        try:
            self.model.set_train(False)
            # self.inversion_trainer.model.noise_level = 0.0
            preds_sample_retokenized = self.embedder_tokenizer(
                decoded_preds,
                padding=True,
                truncation=False,
                return_tensors="ms",
            )["input_ids"]
            preds_sample_retokenized = preds_sample_retokenized[
                : self.args.per_device_eval_batch_size, :
            ]
            pad_token_id = self.pad_token_id
            preds_emb = self.call_embedding_model(
                input_ids=preds_sample_retokenized,
                attention_mask=(preds_sample_retokenized != pad_token_id),
            )
            preds_sample_labels_retokenized = self.embedder_tokenizer(
                decoded_labels, padding=True, truncation=False, return_tensors="ms"
            )["input_ids"]
            preds_sample_labels_retokenized = preds_sample_labels_retokenized[
                : self.args.per_device_eval_batch_size, :
            ]
            labels_emb = self.call_embedding_model(
                input_ids=preds_sample_labels_retokenized,
                attention_mask=(preds_sample_labels_retokenized != pad_token_id),
            )
            emb_cos_sims = ops.cosine_similarity(preds_emb, labels_emb)

            sim_result = {
                "emb_cos_sim": emb_cos_sims.mean().item(),
            }
            self.model.set_train(True)

        except (TypeError, RuntimeError):
            sim_result = {"emb_cos_sim": 0, "emb_cos_sim_sem": 0}

        self.preds_sample_list = preds_sample_list
        self.preds_sample_labels_list = preds_sample_labels_list

        metrics = {**num_tokens_metrics, **bleu_result, **sim_result}
        return metrics

    def evaluation_loop(self, dataset, *args, **kwargs) -> EvalLoopOutput:

        """
        Run evaluation and returns metrics.

        Override to compute ppl from eval loss.
        """

        output = super().evaluation_loop(dataset, *args, **kwargs)
        # metric_key_prefix = kwargs["metric_key_prefix"]
        # # TODO compute some data metrics here too.
        if self.args.local_rank <= 0:
            # Generate some text on worker 0 and compute metrics.
            generation_metrics = self.eval_generation_metrics(dataset)
            output.metrics.update(generation_metrics)
        return output

    #TODO: lack load checkpoint func

    def _remap_state_dict(self, state_dict: Dict) -> Dict:
        """Edit keys posthumously on model load."""
        return state_dict
