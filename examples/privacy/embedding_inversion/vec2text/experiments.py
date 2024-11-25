'''
preprocess data and set experimental procedure
'''
import abc
import functools
import hashlib
import json
import os
import resource
from typing import Dict, Optional
import logging

import numpy as np
import mindspore as ms
from mindspore.dataset import GeneratorDataset
import mindnlp.engine
from mindnlp.transformers.modeling_utils import PreTrainedModel
from mindnlp.transformers import AutoTokenizer
from mindnlp.transformers.tokenization_utils_fast import PreTrainedTokenizer
import datasets  # needed by mindnlp

import aliases
import analyze_utils
from data_helpers import dataset_from_args
from models.config import InversionConfig
from models import CorrectorEncoderModel, InversionModel
from run_args import DataArguments, ModelArguments, TrainingArguments
from tokenize_data import DataCollatorForSeq2Seq, embed_dataset_batch, tokenize_function_, tokenize_function
from utils import dataset_map_single_worker, get_num_proc



# Allow W&B to start slowly.
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["_WANDB_STARTUP_DEBUG"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "False"


# big issues! occasionally found no access to GPU with ms


device = ms.get_context("device_target")
logger = logging.getLogger(__name__)

# We maintain our own cache because huggingface datasets caching
# doesn't always work properly.
DATASET_CACHE_PATH = os.environ.get(
    "VEC2TEXT_CACHE", os.path.expanduser("~/.cache/inversion")
)


def md5_hash_kwargs(**kwargs) -> str:
    # We ignore special hf args that start with _ like '__cached__setup_devices'.
    safe_kwargs = {k: str(v) for k, v in kwargs.items() if not k.startswith("_")}
    s = json.dumps(safe_kwargs, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()


class Experiment(abc.ABC):
    '''
    experiment base class
    '''
    def __init__(self, model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
        # Interactions between args handled here:
        training_args.metric_for_best_model = f"{data_args.dataset_name}_loss"

        logger.info(
            "Save checkpoints according to metric_for_best_model %s:",
            training_args.metric_for_best_model,
        )

        # Save all args.
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        # Set random seed, add hash to output path.
        # transformers.set_seed(training_args.seed)
        mindnlp.engine.set_seed(training_args.seed)


        if training_args.output_dir is None:
            training_args.output_dir = os.path.join("saves", self.kwargs_hash)
        print(f"Experiment output_dir = {training_args.output_dir}")
        # Set up output_dir and wandb.
        self._consider_init_wandb()

    @property
    def config(self) -> InversionConfig:
        return InversionConfig(
            **vars(self.data_args),
            **vars(self.model_args),
            **vars(self.training_args),
        )

    @property
    def is_llama_chat(self) -> bool:
        return self.model_args.embedder_model_name in [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
        ]

    @property
    def dataset_kwargs(self) -> Dict[str, str]:
        return {
            "model_name": self.model_args.model_name_or_path,
            "embedder_name": self.model_args.embedder_model_name,
            "max_seq_length": str(self.model_args.max_seq_length),
            "use_less_data": str(self.data_args.use_less_data),
            "embedder_model_api": str(self.model_args.embedder_model_api),
        }

    def run(self):
        print("----------run start?-------------")
        if self.training_args.do_eval:
            self.evaluate()
        else:
            self.train()

    def train(self) -> Dict:
        '''training'''
        training_args = self.training_args
        logger.info("*** Training ***")
        training_argsdevice = ms.get_context("device_target")
        # Log on each process a small summary of training.
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_argsdevice}, "
            + f"fp16 training: {training_args.fp16}, bf16 training: {training_args.bf16}"
        )
        checkpoint = self._get_checkpoint()

        logging.info("Experiment::train() loaded checkpoint %s", checkpoint)
        trainer = self.load_trainer()
        print(f"train() called – resume-from_checkpoint = {checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        print(metrics)

        trainer.log_metrics("train", metrics)
        # trainer.save_metrics("train", metrics)
        trainer.save_state()
        print("success!!!!!great man!!!")
        return metrics

    def evaluate(self) -> Dict:
        '''Evaluation'''
        logger.info("*** Evaluate ***")
        trainer = self.load_trainer()
        num_eval_samples = len(trainer.eval_dataset)
        metrics = trainer.evaluate()
        max_eval_samples = (
            self.data_args.max_eval_samples
            if self.data_args.max_eval_samples is not None
            else num_eval_samples
        )
        metrics["eval_samples"] = min(max_eval_samples, num_eval_samples)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        return metrics

    def _get_checkpoint(self) -> Optional[str]:
        '''get checkpoint to implement the correction'''
        training_args = self.training_args
        last_checkpoint = None
        if (os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir):
            last_checkpoint = mindnlp.engine.get_last_checkpoint(
                training_args.output_dir
            )
            if (last_checkpoint is None and os.listdir(training_args.output_dir)):
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            if (last_checkpoint is not None and training_args.resume_from_checkpoint is None):
                logger.info(
                    "Checkpoint detected, resuming training at %s. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.",
                    last_checkpoint
                )
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if checkpoint:
            logger.info("Loading from checkpoint %s", checkpoint)
        else:
            logger.info("No checkpoint found, training from scratch")

        print(checkpoint)
        print(last_checkpoint)
        return checkpoint

    @property
    def kwargs_hash(self) -> str:
        all_args = {
            **vars(self.model_args),
            **vars(self.data_args),
            **vars(self.training_args),
        }
        all_args.pop("local_rank")
        # print("all_args:", all_args)
        return md5_hash_kwargs(**all_args)

    @property
    def _world_size(self) -> int:
        #not found in mindspore similar with torch.distributed.get_world_size()
        #TODO: add some distribution function to it
        try:
            return os.environ.get("WORLD_SIZE", 1)
        except (RuntimeError, ValueError):
            return 1

    @property
    def _is_main_worker(self) -> bool:
        return (self.training_args.local_rank <= 0) and (
            int(os.environ.get("LOCAL_RANK", 0)) <= 0
        )

    @property
    @abc.abstractmethod
    def _wandb_project_name(self) -> str:
        raise NotImplementedError()

    @property
    def _wandb_exp_name(self) -> str:
        name_args = [
            self.training_args.exp_group_name,
            self.training_args.exp_name,
            self.model_args.model_name_or_path,
            self.model_args.embedder_model_name,
        ]
        name_args = [n for n in name_args if ((n is not None) and len(n))]
        return "__".join(name_args)

    def _consider_init_wandb(self) -> None:
        '''whether to init wandb'''
        if self.training_args.use_wandb and self._is_main_worker:
            import wandb

            wandb.init(
                project=self._wandb_project_name,
                name=self._wandb_exp_name,
                id=self.kwargs_hash,
                resume=True,
            )
            training_args = vars(self.training_args)
            # deepspeed kwargs are not json serializable
            training_args = {
                k: v for k, v in training_args.items() if "deepspeed" not in k
            }
            wandb.config.update(
                {
                    **vars(self.model_args),
                    **vars(self.data_args),
                    **training_args,
                },
                allow_val_change=True,
            )
            # Long-running experiments have been killed because wandb
            # runs out of file descriptors to write summary files
            # to. Very silly error, but seems unfixed:
            # https://github.com/wandb/wandb/issues/2825
            #
            # Anyway, this line of code should (hopefully) set the
            # limit to infinity so this can't happen.
            resource.setrlimit(
                resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
            )
        else:
            # Disable W&B
            pass
            # os.environ["WANDB_MODE"] = "disabled"
            # os.environ["WANDB_DISABLED"] = "true"

    @abc.abstractmethod
    def load_trainer(self) -> mindnlp.engine.Trainer:
        raise NotImplementedError()

    @abc.abstractmethod
    def load_model(self) -> PreTrainedModel:
        raise NotImplementedError()

    def load_tokenizer(self) -> PreTrainedTokenizer:
        '''load tokenizer'''
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            padding="max_length",
            truncation="max_length",
            max_length=self.model_args.max_seq_length,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Disable super annoying warning:
        # https://github.com/huggingface/transformers/issues/22638
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        return tokenizer

    #lack API(transformers.DataCollatorForSeq2Seq) in mindnlp and achieve it from scratch
    def get_collator(self, tokenizer: PreTrainedTokenizer) -> DataCollatorForSeq2Seq:
        return DataCollatorForSeq2Seq(
            tokenizer,
            model=None,
            label_pad_token_id=-100,
            padding="max_length",
            max_length=self.model_args.max_seq_length,
            pad_to_multiple_of=8 if self.training_args.fp16 else None,
        )

    def _load_train_dataset_uncached(self, tokenizer: AutoTokenizer, embedder_tokenizer: AutoTokenizer):
        '''
        load train dataset uncached
        '''
        data_args = self.data_args
        # Load datasets

        logger.info("Loading dataset '%s'...", self.data_args.dataset_name)
        raw_datasets = dataset_from_args(self.data_args)


        # Remove extra features except for 'frozen_embeddings' which could be embeddings
        # saved to disk.
        # column_names = list(raw_datasets["train"].features)

        column_names = raw_datasets["train"].column_names

        # pylint: disable=C0103
        ALLOWED_COLUMN_NAMES = {"frozen_embeddings"}
        column_names = [c for c in column_names if c not in ALLOWED_COLUMN_NAMES]
        if data_args.use_less_data and (data_args.use_less_data > 0):
            new_length = min(len(raw_datasets["train"]), data_args.use_less_data)
            train_datasets = raw_datasets["train"].take(new_length)
            new_length_ = min(len(raw_datasets["validation"]), data_args.max_eval_samples)
            eval_datasets = raw_datasets["validation"].take(new_length_)



        print(
            ">> using fast tokenizers:", tokenizer.is_fast, embedder_tokenizer.is_fast
        )


        train_datasets = train_datasets.map(tokenize_function(tokenizer, embedder_tokenizer,
                                                              self.model_args.max_seq_length, padding=False),
                                            num_parallel_workers=8)

        #no more proc, some mistakes
        eval_datasets = eval_datasets.map(tokenize_function_(tokenizer, embedder_tokenizer,
                                                             self.model_args.max_seq_length, padding=False),)

        #index_ds = ds.NumpySlicesDataset(np.array(range(train_datasets.get_dataset_size())),\
        #                                 column_names=['idx'])

        #------------------------------val process--------------------------------
        max_eval_samples = min(
            self.data_args.use_less_data, self.data_args.max_eval_samples
        )
        eval_datasets = eval_datasets.take(max_eval_samples)
        #index_ds_ = ds.NumpySlicesDataset(list(range(max_eval_samples)), column_names=['idx'])
        return train_datasets, eval_datasets

    def _prepare_val_datasets_dict(self, model: PreTrainedModel, tokenizer: AutoTokenizer,
                                   embedder_tokenizer: AutoTokenizer, val_datasets_dict: datasets.DatasetDict):
        '''prepare_val_datasets_dict'''
        for name, dataset in val_datasets_dict.items():
            max_eval_samples = min(len(dataset), self.data_args.max_eval_samples)
            val_datasets_dict[name] = val_datasets_dict[name].select(
                range(max_eval_samples)
            )
            val_datasets_dict[name] = val_datasets_dict[name].add_column(
                "idx", range(len(val_datasets_dict[name]))
            )
            val_datasets_dict[name].set_format("ms")

        tokenize_fn = tokenize_function

        for key in val_datasets_dict:
            val_datasets_dict[key] = dataset_map_single_worker(
                dataset=val_datasets_dict[key],
                map_fn=tokenize_fn(
                    tokenizer=tokenizer,
                    embedder_tokenizer=embedder_tokenizer,
                    text_column_name="text",
                    max_seq_length=self.model_args.max_seq_length,
                    padding=False,
                ),
                remove_columns=["text"],
                batched=True,
                batch_size=1024,
                num_proc=get_num_proc(),
                desc="Running tokenizer on dataset",
            )

        # filter out empty examples (these exist for xsum documents).
        val_datasets_dict = val_datasets_dict.filter(lambda ex: ex["length"] > 1)

        if self.model_args.use_frozen_embeddings_as_input:
            # assert torch.cuda.is_available()
            # model = model.to(device)

            new_tokenized_datasets = {}
            for key, d in val_datasets_dict.items():
                new_tokenized_datasets[key] = dataset_map_single_worker(
                    dataset=d,
                    map_fn=functools.partial(embed_dataset_batch, model),
                    batched=True,
                    batch_size=self.training_args.per_device_train_batch_size,
                    # pylint: disable=W0212
                    new_fingerprint=(
                        d._fingerprint + md5_hash_kwargs(**self.dataset_kwargs) + ""
                    ),
                    num_proc=1,
                )
            val_datasets_dict = datasets.DatasetDict(new_tokenized_datasets)
        return val_datasets_dict

    def load_train_and_val_datasets(self, tokenizer: AutoTokenizer,
                                    embedder_tokenizer: AutoTokenizer):
        '''load_train_and_val_datasets'''
        dataset_kwargs: Dict[str, str] = self.dataset_kwargs

        # Only set this if it's true, for backwards-compatibility with
        # when we forgot to cache using this argument.
        if self.model_args.use_frozen_embeddings_as_input:
            dataset_kwargs["use_frozen_embeddings_as_input"] = "True"
            # Deprecated arg below. We used to cache different
            # embeddings for suffixes. Then they became the same.
            # Removing the below line will invalidate other
            # people's caches.
            dataset_kwargs["suffix_conditioning"] = "False"

        # os.environ["TOKENIZERS_PARALLELISM"] = "True"
        print(
            "Loading datasets with TOKENIZERS_PARALLELISM =",
            os.environ.get("TOKENIZERS_PARALLELISM"),
        )
        ######################################################################
        train_dataset_kwargs = {
            "dataset_name": self.data_args.dataset_name,
            **dataset_kwargs,
        }
        train_dataset_path = os.path.join(
            DATASET_CACHE_PATH, (md5_hash_kwargs(**train_dataset_kwargs) + ".npy")
        )
        # Optionally set a train dataset path override
        train_dataset_path = os.environ.get(
            "VEC2TEXT_TRAIN_DATASET_PATH", train_dataset_path
        )
        if os.path.exists(train_dataset_path):
            print("path?", train_dataset_path)
            print("loading train dataset from path:", train_dataset_path)
            train_datasets = datasets.load_from_disk(train_dataset_path)
        else:
            train_datasets, eval_datasets = self._load_train_dataset_uncached(
                tokenizer=tokenizer,
                embedder_tokenizer=embedder_tokenizer,
            )

        #--------------------------------------------
        # i = 0
        # data_list = []
        # for data in train_datasets.create_dict_iterator():
        #     i += 1
        #     data = data['text']
        #     data_list.append(data)
        #     if (i == self.data_args.use_less_data):
        #         break
        # column_names = ['input_ids', 'attention_mask', 'labels', 'length', 'embedder_input_ids',
        #                 'embedder_attention_mask']
        #
        # def data_generator():
        #     for data in data_list:
        #         yield (
        #             data['input_ids'], data['attention_mask'], data['labels'], data['length'][0],
        #             data['embedder_input_ids'],
        #             data['embedder_attention_mask'])
        #
        # train_datasets = GeneratorDataset(data_generator, column_names)
        # --------------------------------------------
        column_names = ['input_ids', 'attention_mask', 'labels', 'length', 'embedder_input_ids',
                        'embedder_attention_mask']

        # create numpy memmap in order to lazy download, but no use, so disgusting bug!
        # filename = '/home/luoyf/vec2text/vec2text/saves/train_dataset/processed_data_' +\
                     #str(self.data_args.use_less_data) + '.dat'
        data_list = []
        u = -1
        # store in memmap
        for data in train_datasets:
            u += 1
            input_ids = data[0]['input_ids'].asnumpy()
            attention_mask = data[0]['attention_mask'].asnumpy()
            labels = data[0]['labels'].asnumpy()
            length = data[0]['length'][0].asnumpy()#为了存储方便扩展成32位
            embedder_input_ids = data[0]['embedder_input_ids'].asnumpy()
            embedder_attention_mask = data[0]['embedder_attention_mask'].asnumpy()
            # idx = np.full(32, data[1].asnumpy())
            combined_array = [input_ids, attention_mask, labels, length, embedder_input_ids, embedder_attention_mask]
            # data_memmap[u] = combined_array
            data_list.append(combined_array)
            if u == self.data_args.use_less_data - 1:
                break
        for i in range(1):
            print(data_list[i])
            print("训练数据格式如上↑")

        def data_generator():
            for i in range(self.data_args.use_less_data):
                yield (
                    ms.Tensor(data_list[i][0].astype(np.int32)),
                    ms.Tensor(data_list[i][1].astype(np.int32)),
                    ms.Tensor(data_list[i][2].astype(np.int32)),
                    ms.Tensor(data_list[i][3].astype(np.int32)),
                    ms.Tensor(data_list[i][4].astype(np.int32)),
                    ms.Tensor(data_list[i][5].astype(np.int32)),
                )
        train_datasets = GeneratorDataset(data_generator, column_names)

        data_list_ = []
        for data in eval_datasets.create_dict_iterator():
            data = data['text']
            data_list_.append(data)
        column_names_ = ['input_ids', 'attention_mask', 'labels', 'length', 'embedder_input_ids',
                         'embedder_attention_mask']

        def data_generator_():
            for data in data_list_:
                yield (
                    data['input_ids'], data['attention_mask'], data['labels'], data['length'][0],
                    data['embedder_input_ids'], data['embedder_attention_mask'])

        eval_datasets = GeneratorDataset(data_generator_, column_names_)
        print("convert success!")

        return (train_datasets, eval_datasets)


class InversionExperiment(Experiment):
    '''
    inversion experiment
    '''
    @property
    def trainer_cls(self):
        return trainers.InversionTrainer

    @property
    def _wandb_project_name(self) -> str:
        return "emb-inv-4"

    def load_model(self) -> PreTrainedModel:
        return InversionModel(
            config=self.config,
        )
    # convert MapDataset with "text" key to GeneratorDataset without it


    def load_trainer(self) -> mindnlp.engine.Trainer:
        model = self.load_model()
        train_dataset, eval_dataset = self.load_train_and_val_datasets(
            tokenizer=model.tokenizer,
            embedder_tokenizer=model.embedder_tokenizer,
        )
        return self.trainer_cls(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # data_collator=self.get_collator(tokenizer=model.tokenizer),
        )


class CorrectorExperiment(Experiment):
    '''
    corrector experiment
    '''
    @property
    def _wandb_project_name(self) -> str:
        return "emb-correct-1"

    def load_trainer(self) -> mindnlp.engine.Trainer:
        if self.training_args.corrector_model_from_pretrained:
            (
                _,
                inversion_trainer,
            ) = analyze_utils.load_experiment_and_trainer_from_pretrained(
                name=self.training_args.corrector_model_from_pretrained,
                # max_seq_length=self.model_args.max_seq_length,
                use_less_data=self.data_args.use_less_data,
            )
        else:
            (
                _,
                inversion_trainer,
            ) = aliases.load_experiment_and_trainer_from_alias(
                alias=self.training_args.corrector_model_alias,
                max_seq_length=self.model_args.max_seq_length,
                use_less_data=self.data_args.use_less_data,
            )
        model = self.load_model()
        return trainers.Corrector(
            model=model,
            inversion_trainer=inversion_trainer,
            args=self.training_args,
            # data_collator=DataCollatorForCorrection(
            #     tokenizer=inversion_trainer.model.tokenizer
            # ),
        )

    def load_model(self) -> PreTrainedModel:
        return CorrectorEncoderModel(
            config=self.config,
        )


EXPERIMENT_CLS_MAP = {
    "inversion": InversionExperiment,
    "corrector": CorrectorExperiment,
    "corrector_encoder": CorrectorExperiment,  # backwards-compatible; does same thing as just 'corrector'
}


def experiment_from_args(model_args, data_args, training_args) -> Experiment:
    if training_args.experiment in EXPERIMENT_CLS_MAP:
        experiment_cls = EXPERIMENT_CLS_MAP[training_args.experiment]  # type: ignore
    else:
        raise ValueError(f"Unknown experiment {training_args.experiment}")
    return experiment_cls(model_args, data_args, training_args)  # type: ignore
