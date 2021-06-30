## BERT Subword Tokenizer for Machine Translation

This repository implements a wrapper code for generating a Wordpiece Vocabulary and BERT Tokenizer model from a dataset using tensorflow-text package. The tokenizers generated with this wrapper script are used in the research article: [Power Law Graph Transformer for Machine Translation and Representation Learning](https://github.com/burcgokden/Power-Law-Graph-Transformer/blob/main/plgt_paper.pdf)

Detailed explanation of subword tokenizer and wordpiece vocabulary generation can be found at [Subword Tokenizers @ tensorflow.org](https://www.tensorflow.org/tutorials/tensorflow_text/subwords_tokenizer) 

#### Key features

- Generates a Wordpiece Vocabulary and BERT Tokenizer from a tensorflow dataset for machine translation.
- Simple interface that takes in all the arguments and generates Vocabulary and Tokenizer model.

#### Sample Run:

Sample run generates Vocabulary and Tokenizer model from tensorflow dataset for PT-EN machine translation task from tensorflow dataset: [ted_hrlr_translate/pt_to_en](https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate#ted_hrlr_translatept_to_en) 

Initialize model parameters for bert vocabulary generator and tokenizer:

```python
import make_vocab_tokenizer as mvt

reserved_tokens= ["[PAD]", "[UNK]", "[START]", "[END]"]
bert_tokenizer_params={"lower_case":True}
bert_vocab_args={
                "vocab_size":15000,
                "reserved_tokens":reserved_tokens,
                "bert_tokenizer_params":bert_tokenizer_params,
                "learn_params":{}
            }
```

 Generate vocabulary and tokenizer model:
 
```python
 make_vocab_tok = mvt.bert_src_tgt_tokenizer(
                 src_lang='pt', 
                 tgt_lang='en',
                 BATCH_SIZE = 1024,
                 dataset_file='ted_hrlr_translate/pt_to_en',
                 train_percent=None,
                 src_vocab_path="./ted_hrlr_translate_pt_vocab.txt",
                 tgt_vocab_path="./ted_hrlr_translate_en_vocab.txt",
                 model_name = "./ted_hrlr_translate_pt_en_tokenizer",
                 load_tokenizer_model=False,
                 make_tokenizer=True,
                 bert_tokenizer_params=bert_tokenizer_params,
                 reserved_tokens=reserved_tokens, 
                 bert_vocab_args=bert_vocab_args
                 ) 
```
