'''
download dataset
'''
import os
from typing import Dict, List
import datasets
from mindnlp.dataset import load_dataset
from run_args import DataArguments


def retain_dataset_columns(d, allowed_columns: List[str]):
    column_names_to_remove = [c for c in d.features if c not in allowed_columns]
    return d.remove_columns(column_names_to_remove)


def load_nq_dpr_corpus()-> datasets.Dataset:
    return load_dataset("jxm/nq_corpus_dpr")


def load_msmarco_corpus():
    # has columns ["title", "text"]. only one split ("train")
    dataset_dict = load_dataset("Tevatron/msmarco-passage-corpus")
    return dataset_dict["train"]


def create_omi_ex(ex: Dict[str, str]) -> Dict[str, str]:
    ex["text"] = ex["user"]
    return ex


def create_ompi_ex(ex: Dict[str, str]) -> Dict[str, str]:
    ex["user"] = ex["user"].strip()
    ex["system"] = ex["system"].strip()
    ex["text"] = ex["system"] + "\n\n" + ex["user"]
    ex["prefix"] = ex["system"] + "\n\n"
    ex["suffix"] = ex["user"]
    return ex


def get_world_size() -> int:
    try:
        return os.environ.get("WORLD_SIZE", 1)
    except (RuntimeError, ValueError):
        return 1



def dataset_from_args(data_args: DataArguments) -> datasets.DatasetDict:
    """Loads a dataset from data_args create in `run_args`."""
    if data_args.dataset_name == "nq":
        raw_datasets = load_nq_dpr_corpus()
        raw_datasets["validation"] = raw_datasets["dev"]
    elif data_args.dataset_name == "msmarco":
        raw_datasets = load_msmarco_corpus()
        raw_datasets = raw_datasets.train_test_split(test_size=0.01)
        raw_datasets["validation"] = raw_datasets["test"]
    else:
        raise ValueError(f"unsupported dataset {data_args.dataset_name}")
    return raw_datasets


def load_ag_news_test():
    return load_dataset("ag_news")["test"]


def load_xsum_val(col: str):
    d = load_dataset("xsum")["validation"]
    d = d.rename_column(col, "text")
    return d


def load_wikibio_val():
    d = load_dataset("wiki_bio")["val"]
    d = d.rename_column("target_text", "text")
    return d


def load_arxiv_val():
    d = load_dataset("ccdv/arxiv-summarization")["validation"]
    d = d.rename_column("abstract", "text")
    return d

def load_anthropic_toxic_prompts():
    d = load_dataset("wentingzhao/anthropic-hh-first-prompt")["train"]
    d = d.rename_column("user", "text")
    return d

def load_python_code_instructions_18k_alpaca():
    d = load_dataset("iamtarun/python_code_instructions_18k_alpaca")["train"]
    d = d.rename_column("instruction", "text")
    return d

def load_standard_val_datasets():
    """Loads a pre-defined set of standard val datasets."""
    d = {
        "ag_news": load_ag_news_test(),
        "anthropic_toxic_prompts": load_anthropic_toxic_prompts(),
        "arxiv": load_arxiv_val(),
        "python_code_alpaca": load_python_code_instructions_18k_alpaca(),
        # "xsum_doc": load_xsum_val("document"),
        # "xsum_summ": load_xsum_val("summary"),
        "wiki_bio": load_wikibio_val(),
    }
    d = {k: retain_dataset_columns(v, ["text"]) for k, v in d.items()}

    return datasets.DatasetDict(d)
