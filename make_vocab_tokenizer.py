
import os
import logging
import time

import tensorflow_datasets as tfds
import tensorflow as tf
import bert_subword_tokenizer as bst

logging.getLogger('tensorflow').setLevel(logging.ERROR)

class bert_src_tgt_tokenizer:
    '''
    Creates a subword vocabulary from dataset and a custom tokenizer object for source and
    target languages.
    Default inputs are for portuguese to english dataset from TED Talks Open Translation Project for 15k Vocabulary.
    '''
    def __init__(self,
                 src_lang='pt',
                 tgt_lang='en',
                 BATCH_SIZE = 1024,
                 dataset_file='ted_hrlr_translate/pt_to_en',
                 train_percent=None, #a percentage from 1-100
                 src_vocab_path="./ted_hrlr_translate_pt_vocab.txt",
                 tgt_vocab_path="./ted_hrlr_translate_en_vocab.txt",
                 model_path = "./ted_hrlr_translate_pt_en_tokenizer",
                 load_tokenizer_model=False,
                 make_tokenizer=True,
                 bert_tokenizer_params=None,
                 reserved_tokens=None,
                 bert_vocab_args=None
                 ):
        '''
        Arguments:
            src_lang: source language abbreviation as string. Default is 'pt' for portuguese.
            tgt_lang: target language abbreviation as string. Default is 'en' for english.
            BATCH_SIZE: batch size for dataset.
            dataset_file: location to tensorflow dataset on disk to load. Default is for ted_hrlr_datese for pt-en
            train_percent: set percentage to load from dataset. Default is None (100%).
            src_vocab_path:Path to source vocabulary file to be created.
            tgt_vocab_path: Path to target vocabulary file to be created.
            model_path: tokenizer model path location to save under or load from.
            load_tokenizer_model: If True loads tokenizer model at model_path path. Dataset is not used to create
                                  vocabulary and tokenizer. Default is False.
            make_tokenizer: If True tokenizer is created from vocabulary and saved at model_path. If False only vocabulary is created
                            from dataset. Default is True.
            bert_tokenizer_params: parameter dict for bert tokenizer. None to set default values.
            reserved_tokens: reserved tokenizer used for bert tokenizer. None to set default values.
            bert_vocab_args: parameter dict to create vocabulary from dataset. None to set default values.
        Returns:
            BERT subword tokenizer object for source and target languages.
        '''

        self.load_tokenizer_model=load_tokenizer_model
        self.model_path = model_path
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.make_tokenizer=make_tokenizer

        if self.load_tokenizer_model:
            #load tokenizer model only from model_path.
            print("TOKENIZER INITIALIZED FROM SAVED MODEL")
            #load model into tokenizers object for lang1(pt) and lang2(en)
            self.tokenizers=tf.saved_model.load(self.model_path)
            print([item for item in dir(getattr(self.tokenizers, self.src_lang, None)) if not item.startswith('_')])
            print([item for item in dir(getattr(self.tokenizers, self.tgt_lang, None)) if not item.startswith('_')])
        else:
            #prepare dataset, create vocabularies, create tokenizers and save at model_path.
            self.BATCH_SIZE = BATCH_SIZE
            self.src_vocab_path = src_vocab_path
            self.tgt_vocab_path = tgt_vocab_path

            if reserved_tokens is not None:
                self.reserved_tokens = reserved_tokens
            else:
                self.reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

            if bert_tokenizer_params is not None:
                self.bert_tokenizer_params = bert_tokenizer_params
            else:
                self.bert_tokenizer_params = {"lower_case": True}

            if bert_vocab_args is not None:
                self.bert_vocab_args = bert_vocab_args
            else:
                self.bert_vocab_args = {
                    "vocab_size": 15000,
                    "reserved_tokens": self.reserved_tokens,
                    "bert_tokenizer_params": self.bert_tokenizer_params,
                    "learn_params": {}
                }

            # create vocabulary generator objects
            self.src_vocab_obj = bst.gen_bert_vocab(bert_tokenizer_params, reserved_tokens, bert_vocab_args)
            self.tgt_vocab_obj = bst.gen_bert_vocab(bert_tokenizer_params, reserved_tokens, bert_vocab_args)

            print("LOADING DATASET")
            if train_percent:
                #load only percentage of train data
                examples, metadata = tfds.load(dataset_file,
                                                split=[f"train[:{train_percent}%]", 'validation', 'test'],
                                                with_info=True, as_supervised=True)
            else:
                #load all data
                examples, metadata = tfds.load(dataset_file, split=["train", 'validation', 'test'], with_info=True, as_supervised=True)


            self.train_examples = examples[0]
            self.val_examples = examples[1]
            self.test_examples=examples[2]
            self.metadata=metadata

            print("TOKENIZER INITIALIZED FOR CREATION FROM VOCABULARY")
            self.tokenizers = tf.Module()  # intialize tokenizer for model

            #create monolingual datasets for vocabulary creation
            print("CREATING MONO DATASETS FOR VOCABULARY GENERATION")
            self.train_batches_src, self.train_batches_tgt=self.src_tgt_conv_mono(self.train_examples)
            print("MONO DATASETS CREATED")

            print("MAKING VOCABULARIES")
            self.src_tgt_make_vocab()
            print("VOCABULARIES DONE")

            if self.make_tokenizer:
                print("MAKING TOKENIZER")
                self.src_tgt_make_tokenizer()
                print("TOKENIZER DONE")

    def src_tgt_make_vocab(self):
        '''
        Method to create vocabulary text files from dataset.
        Returns: source and target vocabularies as text files and path locations.
        '''

        if self.load_tokenizer_model:
            print(f"Tokenizer model is loaded from {self.model_path}")
            self.src_vocab_path, self.tgt_vocab_path = None, None
        else:
            if self.src_vocab_path:
                print(f"Creating  {self.src_lang} vocabulary for tokenizer at {self.src_vocab_path}")
                start=time.time()
                self.src_vocab_obj.generate_vocab(self.train_batches_src, filepath=self.src_vocab_path)
                print(f"{self.src_lang} vocabulary done in {time.time() - start:.2f} s")
            if self.tgt_vocab_path:
                print(f"Creating {self.tgt_lang} vocabulary for tokenizer at {self.tgt_vocab_path}")
                start=time.time()
                self.tgt_vocab_obj.generate_vocab(self.train_batches_tgt, filepath=self.tgt_vocab_path) #define train_batches for single sentences.
                print(f"{self.tgt_lang} vocabulary done in {time.time() - start:.2f} s")

        return self.src_vocab_path, self.tgt_vocab_path

    def src_tgt_conv_mono(self, ds):
        '''
        Converts a dataset to monolingual components for vocabulary file creation.
        Arguments:
            ds: tensorflow dataset with src and tgt sentence pairs.
        Returns:
            ds_src: dataset consisting of source langugaes.
            ds_tgt: dataset consisting of target languages.
        '''
        if self.load_tokenizer_model:
            print(f"Tokenizer model is loaded from {self.model_path}")
            ds_src, ds_tgt = None, None
        else:
            ds_src=ds.map(lambda src, tgt: src)
            ds_src=ds_src.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

            ds_tgt=ds.map(lambda src, tgt: tgt)
            ds_tgt=ds_tgt.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        return ds_src, ds_tgt


    def src_tgt_make_tokenizer(self):
        '''
        Generate bert tokenizer using the vocabulary from dataset.
        Returns:
            Tokenizer model and path to tokenizer model.
        '''

        if self.load_tokenizer_model:
            print(f"Tokenizer model is loaded from {self.model_path}")
        else:
            if os.path.isfile(os.path.abspath(self.src_vocab_path)):
                print(f"Creating tokenizer for {self.src_lang}")
                start=time.time()
                setattr(self.tokenizers,self.src_lang, bst.CustomTokenizer(self.reserved_tokens, self.src_vocab_path))
                print(f"en tokenizer done in {time.time()-start:.2f} s")
            else:
                print(f"No vocab file for {self.src_lang}, tokenizer not created")

            if os.path.isfile(os.path.abspath(self.tgt_vocab_path)):
                print(f"Creating tokenizer for {self.tgt_lang}")
                start=time.time()
                setattr(self.tokenizers, self.tgt_lang, bst.CustomTokenizer(self.reserved_tokens, self.tgt_vocab_path))
                print(f"de tokenizer done in {time.time() - start:.2f} s")
            else:
                print(f"No vocab file for {self.tgt_lang}, tokenizer not created")

            if self.model_path:
                print(f"Saving tokenizer as saved model at {os.path.abspath(self.model_path)}")
                tf.saved_model.save(self.tokenizers, os.path.abspath(self.model_path))

        return self.tokenizers, self.model_path
