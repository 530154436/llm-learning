#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/4/27 19:29
# @function:
from typing import Dict, List
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer


def train(
    input_file: str,
    vocab_size: int,
    model_name: str,
    model_type: str,
    character_coverage: float
):
    """
    search on https://github.com/google/sentencepiece/blob/master/doc/options.md to learn more about the parameters
    :param input_file: one-sentence-per-line raw corpus file. No need to run tokenizer, normalizer or preprocessor.
                       By default, SentencePiece normalizes the input with Unicode NFKC.
                       You can pass a comma-separated list of files.
    :param vocab_size: vocabulary size, e.g., 8000, 16000, or 32000
    :param model_name: output model name prefix. <model_name>.model and <model_name>.vocab are generated.
    :param model_type: model type. Choose from unigram (default), bpe, char, or word.
                       The input sentence must be pretokenized when using word type.
    :param character_coverage: amount of characters covered by the model, good defaults are: 0.9995 for languages with
                               rich character set like Japanse or Chinese and 1.0 for other languages with
                               small character set.
    """
    input_argument = '--input=%s --model_prefix=%s --vocab_size=%s --model_type=%s --character_coverage=%s ' \
                     '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
    cmd = input_argument % (input_file, model_name, vocab_size, model_type, character_coverage)
    SentencePieceTrainer.Train(cmd)


def run():
    en_input = 'data/news-commentary-v13.zh-en.en'
    en_vocab_size = 32000
    en_model_name = 'spm_en_bpe'
    en_model_type = 'bpe'
    en_character_coverage = 1
    train(en_input, en_vocab_size, en_model_name, en_model_type, en_character_coverage)

    ch_input = 'data/news-commentary-v13.zh-en.zh'
    ch_vocab_size = 32000
    ch_model_name = 'spm_zh_bpe'
    ch_model_type = 'bpe'
    ch_character_coverage = 0.9995
    train(ch_input, ch_vocab_size, ch_model_name, ch_model_type, ch_character_coverage)


class SentencePieceTokenizerWithLang:
    """
    分词器(Sentencepiece)
    特殊Token: <UNK> <PAD> <BOS> <EOS> 分别对应 0 1 2 3
    """
    tokenizers: Dict[str, SentencePieceProcessor] = dict()

    def __init__(self, lang: str):
        self.lang = lang
        self.tokenizer = self.init(lang)

    @classmethod
    def init(cls, lang: str):
        """ 加载 sentencepiece tokenizer
        """
        if lang in cls.tokenizers:
            return cls.tokenizers[lang]
        tokenizer = SentencePieceProcessor()
        tokenizer.Load(f"data/spm/spm_{lang}_bpe.model")
        cls.tokenizers[lang] = tokenizer
        return tokenizer

    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size()

    def unk_id(self) -> int:
        return self.tokenizer.unk_id()

    def pad_id(self) -> int:
        return self.tokenizer.pad_id()

    def bos_id(self) -> int:
        return self.tokenizer.bos_id()

    def eos_id(self) -> int:
        return self.tokenizer.eos_id()

    def encode_as_token(self, text: str) -> str:
        return self.tokenizer.EncodeAsPieces(text)

    def encode_as_id(self, text: str) -> List[int]:
        return self.tokenizer.EncodeAsIds(text)

    def decode_ids(self, token_ids: List[int]) -> str:
        return self.tokenizer.DecodeIds(token_ids)


def test():
    sp = SentencePieceTokenizerWithLang(lang='zh')
    text = "美国总统特朗普今日抵达夏威夷。"

    print(sp.encode_as_token(text))
    print(sp.encode_as_id(text))
    a = [2, 12908, 277, 7420, 7319, 18385, 28724, 3]
    print(sp.decode_ids(a))
    print(sp.vocab_size())


if __name__ == "__main__":
    # run()
    test()
