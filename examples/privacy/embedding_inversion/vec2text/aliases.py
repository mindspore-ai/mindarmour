'''
get zero step model arguments setup for the second phrase of embedding inversion(corrector)
'''
import analyze_utils

# TODO always load args from disk, delete this dict.
ARGS_DICT = {
    "gtr_nq__msl128_beta": (
        "--dataset_name nq "
        "--per_device_train_batch_size 128 "
        "--per_device_eval_batch_size 128 "
        "--max_seq_length 128 "
        "--model_name_or_path t5-base "
        "--embedder_model_name gtr_base "
        "--num_repeat_tokens 16 "
        "--embedder_no_grad True "
        "--exp_group_name mar17-baselines "
        "--learning_rate 0.0003 "
        "--freeze_strategy none "
        "--embedder_fake_with_zeros False "
        "--use_frozen_embeddings_as_input False "
        "--num_train_epochs 24 "
        "--max_eval_samples 500 "
        "--eval_steps 25000 "
        "--warmup_steps 100000 "
        "--bf16=1 "
        "--use_wandb=0"
    ),
    "paraphrase_nq__msl32__10epoch": (
        "--per_device_train_batch_size 128 "
        "--per_device_eval_batch_size 128 "
        "--max_seq_length 32 "
        "--model_name_or_path google-t5/t5-base "
        "--dataset_name nq "
        "--embedder_model_name gtr_base "
        "--num_repeat_tokens 16 "
        "--embedder_no_grad True "
        "--num_train_epochs 1 "
        "--max_eval_samples 16 "
        "--eval_steps 400 "
        "--warmup_steps 300 "
        "--bf16 1 "
        "--use_frozen_embeddings_as_input False "
        "--experiment inversion "
        "--learning_rate 0.001 "
        "--output_dir ./saves/gtr-XXXxxx "
        "--save_steps 10000000000 "
        "--use_less_data 2560"
    )
}


# Dictionary mapping model names
CHECKPOINT_FOLDERS_DICT = {
    ############################# MSMARCO ##############################
    "paraphrase_nq__msl32__10epoch": "/home/luoyf/vec2text/vec2text/saves/gtr-X",
}


def load_experiment_and_trainer_from_alias(alias: str, max_seq_length: int = None, use_less_data: int = None):
    """
    Load the experimental setup and corresponding trainer based on a given alias.

    Parameters:
    alias (str): The identifier used to select the experiment setup.
    max_seq_length (int, optional): The maximum sequence length for the model. Defaults to None.
    use_less_data (int, optional): A flag to indicate if a reduced dataset should be used. Defaults to None.

    Returns:
    type: Description of the return value (if applicable)
    """
    try:
        args_str = ARGS_DICT.get(alias)
        checkpoint_folder = CHECKPOINT_FOLDERS_DICT[alias]
        print("-----------args_str的值是---------------")
        print(args_str)

    except KeyError:
        print(f"{alias} not found in aliases.py, using as checkpoint folder")
        args_str = None
        checkpoint_folder = alias
    print(f"loading alias {alias} from {checkpoint_folder}...")
    experiment, trainer = analyze_utils.load_experiment_and_trainer(
        checkpoint_folder,
        args_str,
        do_eval=False,
        max_seq_length=max_seq_length,
        use_less_data=use_less_data,
    )
    return experiment, trainer


def load_model_from_alias(alias: str, max_seq_length: int = None):
    _, trainer = load_experiment_and_trainer_from_alias(
        alias, max_seq_length=max_seq_length
    )
    return trainer.model
