# coding: utf-8
"""
Data module
"""
import sys
import random
import os
import os.path
from typing import Optional
import logging

import torch
from torchtext import data
from joeynmt.translation import TranslationDataset
# from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field

from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from joeynmt.vocabulary import build_vocab, Vocabulary

logger = logging.getLogger(__name__)


def load_data(data_cfg: dict, datasets: list = None)\
        -> (Dataset, Dataset, Optional[Dataset], Vocabulary, Optional[Vocabulary]):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    If you set ``random_train_subset``, a random selection of this size is used
    from the training set instead of the full training set.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :param datasets: list of dataset names to load
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: source vocabulary extracted from training data, 
            `None` if not continuous src features
        - trg_vocab: target vocabulary extracted from training data
    """
    if datasets is None:
        datasets = ["train", "dev", "test"]

    # load data from files
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    train_path = data_cfg.get("train", None)
    dev_path = data_cfg.get("dev", None)
    test_path = data_cfg.get("test", None)

    if train_path is None and dev_path is None and test_path is None:
        raise ValueError('Please specify at least one data source path.')

    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    tok_fun = lambda s: list(s) if level == "char" else s.split()
    
    # WIP: Cihan's implementation
    # TODO: refactor, check if using Field.sequential=False would work better?
    def tok_fun_cont(features):
        ft_list = torch.split(features, 1, dim=0)
        return [ft.squeeze() for ft in ft_list]

    # TODO: fix using **kwargs
    # NOTE (Cihan): The something was necessary to match the function signature.
    def stack_features(features, something):
        return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)
    
    if data_cfg["continuous_src_features"]:
        # src_field = data.Field(sequential=False, use_vocab=False,
        #                     init_token=None, eos_token=EOS_TOKEN,
        #                     pad_token=PAD_TOKEN, tokenize=tok_fun,
        #                     batch_first=True, lower=lowercase,
        #                     unk_token=UNK_TOKEN,
        #                     include_lengths=True)
        src_field = data.Field(sequential=True, use_vocab=False,
                            dtype=torch.float32, preprocessing=tok_fun_cont,
                            init_token=None, eos_token=None,
                            pad_token=torch.zeros((data_cfg["input_size"],)), # tokenize=lambda x: x,
                            batch_first=True, lower=False,
                            postprocessing=stack_features,
                            include_lengths=True)
        logger.info("   ** src field **")
        logger.info("SRC FIELD IS:  %s", src_field)
    else:
        src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                            pad_token=PAD_TOKEN, tokenize=tok_fun,
                            batch_first=True, lower=lowercase,
                            unk_token=UNK_TOKEN,
                            include_lengths=True)

    trg_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           unk_token=UNK_TOKEN,
                           batch_first=True, lower=lowercase,
                           include_lengths=True)

    train_data = None
    if "train" in datasets and train_path is not None:
        logger.info("loading training data...")
        
        if data_cfg["continuous_src_features"]:
            train_data = TranslationDataset(cont_input_features=True,
                                            path=train_path,
                                            exts=("." + src_lang, "." + trg_lang),
                                            fields=(src_field, trg_field),
                                            filter_pred=
                                            lambda x: len(vars(x)['src']) # filter_pred (callable or None): Use only examples for which filter_pred(example) is True, or use all examples if None.
                                            <= max_sent_length
                                            and len(vars(x)['trg'])
                                            <= max_sent_length)

        else:
            train_data = TranslationDataset(cont_input_features=False,
                                            path=train_path,
                                            exts=("." + src_lang, "." + trg_lang),
                                            fields=(src_field, trg_field),
                                            filter_pred=
                                            lambda x: len(vars(x)['src']) # filter_pred (callable or None): Use only examples for which filter_pred(example) is True, or use all examples if None.
                                            <= max_sent_length
                                            and len(vars(x)['trg'])
                                            <= max_sent_length)            




        random_train_subset = data_cfg.get("random_train_subset", -1)
        if random_train_subset > -1:
            # select this many training examples randomly and discard the rest
            keep_ratio = random_train_subset / len(train_data)
            keep, _ = train_data.split(
                split_ratio=[keep_ratio, 1 - keep_ratio],
                random_state=random.getstate())
            train_data = keep

    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    if not data_cfg["continuous_src_features"]:
        src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
        src_min_freq = data_cfg.get("src_voc_min_freq", 1)
        src_vocab_file = data_cfg.get("src_vocab", None) 
        assert (train_data is not None) or (src_vocab_file is not None)

    trg_vocab_file = data_cfg.get("trg_vocab", None)
    assert (train_data is not None) or (trg_vocab_file is not None)

    src_vocab = None
    if not data_cfg["continuous_src_features"]:
        logger.info("building src vocabulary...")

        src_vocab = build_vocab(field="src", min_freq=src_min_freq,
                                max_size=src_max_size,
                                dataset=train_data, vocab_file=src_vocab_file)

    logger.info("building trg vocabulary...")
    trg_vocab = build_vocab(field="trg", min_freq=trg_min_freq,
                            max_size=trg_max_size,
                            dataset=train_data, vocab_file=trg_vocab_file)


    dev_data = None
    if "dev" in datasets and dev_path is not None:
        logger.info("loading dev data...")

        if data_cfg["continuous_src_features"]:
            dev_data = TranslationDataset(cont_input_features=True,
                                          path=dev_path,
                                          exts=("." + src_lang, "." + trg_lang),
                                          fields=(src_field, trg_field))
        else:
            dev_data = TranslationDataset(cont_input_features=False,
                                          path=dev_path,
                                          exts=("." + src_lang, "." + trg_lang),
                                          fields=(src_field, trg_field))
           
            
    test_data = None
    if "test" in datasets and test_path is not None:
        logger.info("loading test data...")
        
        if data_cfg["continuous_src_features"]:
            test_data = TranslationDataset(cont_input_features=True,
                                           path=test_path,
                                           exts=("." + src_lang, "." + trg_lang),
                                           fields=(src_field, trg_field))
    
        else:
            # check if target exists
            if os.path.isfile(test_path + "." + trg_lang):
                test_data = TranslationDataset(cont_input_features=False,
                    path=test_path, exts=("." + src_lang, "." + trg_lang),
                    fields=(src_field, trg_field))
            else:
                # no target is given -> create dataset from src only
                test_data = MonoDataset(path=test_path, ext="." + src_lang,
                                        field=src_field)
    
    if not data_cfg["continuous_src_features"]:
        logger.info("src vocab len: %d\n", len(src_vocab))
        logger.info("src vocab: %s\n", src_vocab)

        src_field.vocab = src_vocab
    
    trg_field.vocab = trg_vocab
    logger.info("data loaded.")
    return train_data, dev_data, test_data, src_vocab, trg_vocab


# pylint: disable=global-at-module-level
global max_src_in_batch, max_tgt_in_batch


# pylint: disable=unused-argument,global-variable-undefined
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    src_elements = count * max_src_in_batch
    if hasattr(new, 'trg'):  # for monolingual data sets ("translate" mode)
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
        tgt_elements = count * max_tgt_in_batch
    else:
        tgt_elements = 0
    return max(src_elements, tgt_elements)


def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   batch_type: str = "sentence",
                   train: bool = False,
                   shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=shuffle)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.BucketIterator(
            repeat=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=False, sort=False)

    return data_iter


class MonoDataset(Dataset):
    """Defines a dataset for machine translation without targets."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path: str, ext: str, field: Field, **kwargs) -> None:
        """
        Create a monolingual dataset (=only sources) given path and field.

        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param field: Containing the fields that will be used for data.
        :param kwargs: Passed to the constructor of data.Dataset.
        """

        fields = [('src', field)]

        if hasattr(path, "readline"):  # special usage: stdin
            src_file = path
        else:
            src_path = os.path.expanduser(path + ext)
            src_file = open(src_path)

        examples = []
        for src_line in src_file:
            src_line = src_line.strip()
            if src_line != '':
                examples.append(data.Example.fromlist(
                    [src_line], fields))

        src_file.close()

        super().__init__(examples, fields, **kwargs)
