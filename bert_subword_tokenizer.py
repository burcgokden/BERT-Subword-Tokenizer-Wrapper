
import os
import pathlib
import re

import tensorflow_text as text
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
pwd = pathlib.Path.cwd()

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

class gen_bert_vocab():
    '''
    This class is a wrapper for generating vocabulary from dataset from subword tokenizer tutorial at
    https://www.tensorflow.org/tutorials/tensorflow_text/subwords_tokenizer
    https://github.com/tensorflow/text/blob/master/examples/subwords_tokenizer.ipynb

    Generates a subword vocabulary using BERT subword vocabulary generator.
    '''
    def __init__(self, bert_tokenizer_params=None, reserved_tokens=None, bert_vocab_args=None):
        '''
        Arguments:
            bert_tokenizer_params: dictionary. Parameters provided for BERT tokenizer.
            reserved_tokens: list of string. reserved tokens for BERT tokenizer.
            bert_vocab_args: dictionary. Parameters provided to BERT vocabulary generator from dataset
        Returns:
            gen_bert_vocab class object for vocabulary generation.
        '''


        if bert_tokenizer_params:
            self.bert_tokenizer_params= bert_tokenizer_params
        else:
            self.bert_tokenizer_params={"lower_case":True}

        if reserved_tokens:
            self.reserved_tokens=reserved_tokens
        else:
            self.reserved_tokens= ["[PAD]", "[UNK]", "[START]", "[END]"]

        if bert_vocab_args:
            self.bert_vocab_args=bert_vocab_args
        else:
            self.bert_vocab_args={
                "vocab_size":15000,
                "reserved_tokens":self.reserved_tokens,
                "bert_tokenizer_params":self.bert_tokenizer_params,
                "learn_params":{}
            }

    def generate_vocab(self, ds, filepath=None):
        '''
        Generates vocabulary from tensorflow dataset
        Arguments:
            ds: A tensorflow dataset containing monolingual sentences.
            filepath: a file path to save vocabulary as text file
        Returns:
            A vocabulary and saves text file containing vocabulary from dataset.
        '''

        vocab=bert_vocab.bert_vocab_from_dataset(ds, **self.bert_vocab_args)

        if filepath:
            print(f"writing vocabulary to file: {os.path.abspath(filepath)}")
            self.write_vocab_file(os.path.abspath(filepath), vocab)
        else:
            print("Skipped writing vocabulary to file")

        return vocab

    @staticmethod
    def write_vocab_file(filepath, vocab):
        '''
        Write vocabulary to file one token at a time.
        Arguments:
            filepath: text file path for vocabulary.
            vocab: vocabulary generated from dataset.
        Returns:
            A text file containing vocabulary.
        '''
        with open(os.path.abspath(filepath), 'w') as f:
            for token in vocab:
                print(token, file=f)


class CustomTokenizer(tf.Module):
    '''
    A custom BERT tokenizer module class wrapper from subword tutorial at
    https://www.tensorflow.org/tutorials/tensorflow_text/subwords_tokenizer
    https://github.com/tensorflow/text/blob/master/examples/subwords_tokenizer.ipynb
    '''

    def __init__(self, reserved_tokens, vocab_path, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)

        ## Create the signatures for export:

        # Include a tokenize signature for a batch of strings.
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        # Merge the `word` and `word-piece` axes.
        enc = enc.merge_dims(-2, -1)
        enc = self.add_start_end(enc)
        return enc

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return self.cleanup_text(self._reserved_tokens, words)

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)

    def add_start_end(self, ragged):
        START = tf.argmax(tf.constant(self._reserved_tokens) == "[START]")
        END = tf.argmax(tf.constant(self._reserved_tokens) == "[END]")
        count = ragged.bounding_shape()[0]
        starts = tf.fill([count, 1], START)
        ends = tf.fill([count, 1], END)
        return tf.concat([starts, ragged, ends], axis=1)

    @staticmethod
    def cleanup_text(reserved_tokens, token_txt):
        # Drop the reserved tokens, except for "[UNK]".
        bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
        bad_token_re = "|".join(bad_tokens)

        bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
        result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

        # Join them into strings.
        result = tf.strings.reduce_join(result, separator=' ', axis=-1)

        return result
