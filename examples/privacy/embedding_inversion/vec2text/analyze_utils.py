'''
load experiment and trainer
'''
import argparse
import os
import glob
import json
import shlex
from typing import Optional


import pandas as pd
from mindnlp.engine import get_last_checkpoint
import mindnlp.utils.logging as logging
import mindspore as ms

import experiments
from models.config import InversionConfig
from run_args import DataArguments, ModelArguments, TrainingArguments, \
    parse_args_into_dataclasses


# from mindnlp.accelerate import PartialState
# import error, can't find this package.

# no need for data transformation across multiple device
# def set_device_context():
#     try:
#         # 尝试设置为 GPU
#         context.set_context(device_target="GPU")
#         print("Using GPU")
#     except:
#         try:
#             # 如果 GPU 不可用，尝试设置为 Ascend
#             context.set_context(device_target="Ascend")
#             print("Using Ascend")
#         except:
#             # 如果 Ascend 也不可用，使用 CPU
#             context.set_context(device_target="CPU")
#             print("Using CPU")
# set_device_context()

# device = torch.device(
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )

logging.set_verbosity_error()


#corrector 的第二个阶段的两次加载都调用这个了
def load_experiment_and_trainer(
        checkpoint_folder: str,
        args_str: Optional[str] = None,
        checkpoint: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        use_less_data: Optional[int] = None,
):
    '''
    (can't import due to circular import) -> trainers.InversionTrainer:
    import previous aliases so that .bin that were saved prior to the
    existence of the vec2text module will still work.
    '''

    if checkpoint is None:
        checkpoint = get_last_checkpoint(checkpoint_folder)  # a checkpoint
    if checkpoint is None:
        # This happens in a weird case, where no model is saved to saves/xxx/checkpoint-*/pytorch_model.bin
        # because checkpointing never happened (likely a very short training run) but there is still a file
        # available in saves/xxx/pytorch_model.bin.
        checkpoint = checkpoint_folder
    print("Loading model from checkpoint:", checkpoint)

    if args_str is not None:
        #先后有两次args，第一次是command line中的args，还有一次是调用的写入alias中的明文args

        args_list = shlex.split(args_str) # not namespace format which can be tackled with identical operation like the first call

        parser = argparse.ArgumentParser()
        for i in range(0, len(args_list) - 1, 2):
            arg_name = args_list[i].lstrip('-')
            arg_value = args_list[i + 1]

            try:
                arg_value = int(arg_value)
            except ValueError:
                if arg_value == 'True':
                    arg_value = True
                elif arg_value == 'False':
                    arg_value = False
            parser.add_argument(f'--{arg_name}', default=arg_value, type=type(arg_value))

        args = parser.parse_args(args_list)

        # traing_args may not be a normal dataclass, and then should be adapted to the new one.
        model_args, data_args, training_args = parse_args_into_dataclasses(args)
    else:
        try:
            data_args = ms.load_checkpoint(os.path.join(checkpoint, os.pardir, "data_args.bin"))
        except FileNotFoundError:
            data_args = ms.load_checkpoint(os.path.join(checkpoint, "data_args.bin"))
        try:
            model_args = ms.load_checkpoint(
                os.path.join(checkpoint, os.pardir, "model_args.bin")
            )
        except FileNotFoundError:
            model_args = ms.load_checkpoint(os.path.join(checkpoint, "model_args.bin"))
        try:
            training_args = ms.load_checkpoint(
                os.path.join(checkpoint, os.pardir, "training_args.bin")
            )
        except FileNotFoundError:
            training_args = ms.load_checkpoint(os.path.join(checkpoint, "training_args.bin"))

    training_args.dataloader_num_workers = 0  # no multiprocessing :)
    training_args.use_wandb = False
    training_args.report_to = []
    training_args.mock_embedder = False
    # training_args.no_cuda = not (context.get_context("device_target")=="GPU")

    if max_seq_length is not None:
        print(
            f"Overwriting max sequence length from {model_args.max_seq_length} to {max_seq_length}"
        )
        model_args.max_seq_length = max_seq_length

    if use_less_data is not None:
        print(
            f"Overwriting use_less_data from {data_args.use_less_data} to {use_less_data}"
        )
        data_args.use_less_data = use_less_data

    experiment = experiments.experiment_from_args(model_args, data_args, training_args)
    trainer = experiment.load_trainer()
    # pylint: disable=W0212
    trainer.model._keys_to_ignore_on_save = []
    try:
        # pylint: disable=W0212
        trainer._load_from_checkpoint(checkpoint)
    except RuntimeError:
        # backwards compatibility from adding/removing layernorm
        trainer.model.use_ln = False
        trainer.model.layernorm = None
        # try again without trying to load layernorm
        # pylint: disable=W0212
        trainer._load_from_checkpoint(checkpoint)
    return experiment, trainer


def load_trainer(
        *args, **kwargs
):  # (can't import due to circluar import) -> trainers.Inversion
    _, trainer = load_experiment_and_trainer(*args, **kwargs)
    return trainer


def load_results_from_folder(name: str) -> pd.DataFrame:
    filenames = glob.glob(os.path.join(name, "*.json"))
    data = []
    for f in filenames:
        d = json.load(open(f, "r"))
        if "_eval_args" in d:
            # unnest args for evaluation
            d.update(d.pop("_eval_args"))
        data.append(d)
    return pd.DataFrame(data)


def args_from_config(args_cls, config):
    args = args_cls()
    for key, value in vars(config).items():
        if key in dir(args):
            setattr(args, key, value)
    return args


def load_experiment_and_trainer_from_pretrained(name: str, use_less_data: int = 1000):
    '''load experiment and trainer from pretrained model'''

    config = InversionConfig.from_pretrained(name)
    model_args = args_from_config(ModelArguments, config)
    data_args = args_from_config(DataArguments, config)
    training_args = args_from_config(TrainingArguments, config)

    data_args.use_less_data = use_less_data
    training_args.bf16 = 0  # no bf16 in case no support from GPU
    training_args.local_rank = -1  # Don't load in DDP

    training_args.deepspeed_plugin = None  # For backwards compatibility
    training_args.use_wandb = False
    training_args.report_to = []
    training_args.mock_embedder = False
    training_args.output_dir = "saves/" + name.replace("/", "__")


    experiment = experiments.experiment_from_args(model_args, data_args, training_args)
    trainer = experiment.load_trainer()
    trainer.model = trainer.model.__class__.from_pretrained(name)
    # trainer.model.to(training_args.device)
    return experiment, trainer
