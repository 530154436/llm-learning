#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from typing import List
from torch.nn.utils.rnn import pad_sequence
from modules.mask import create_padding_mask, create_sequence_mask, create_casual_mask
from modules.models import Transformer
from translation import config
from translation.sentencepiece_tokenizer import SentencePieceTokenizerWithLang


SRC_TOKENIZER = SentencePieceTokenizerWithLang(lang="en")
TGT_TOKENIZER = SentencePieceTokenizerWithLang(lang="zh")
MODEL = Transformer(src_vocab_size=SRC_TOKENIZER.vocab_size(),
                    tgt_vocab_size=TGT_TOKENIZER.vocab_size(),
                    d_model=config.d_model, num_heads=config.n_heads, d_ff=config.d_ff,
                    dropout=config.dropout, N=config.n_layers)
MODEL.load_state_dict(torch.load(config.model_path, weights_only=True))
MODEL.eval()

@torch.no_grad()
def translate(sentence: str,
              model: Transformer,
              max_len=80):
    """
    模型测试
    """
    src_token_ids = [[SRC_TOKENIZER.bos_id()] + SRC_TOKENIZER.encode_as_id(sentence) + [SRC_TOKENIZER.eos_id()]]
    src_tensor = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_token_ids],
                              batch_first=True, padding_value=SRC_TOKENIZER.pad_id())
    src_mask = create_padding_mask(src_tensor, pad_token_id=SRC_TOKENIZER.pad_id())
    # print(src_tensor.shape, src_mask.shape)
    # print(src_tensor)
    # print(src_mask)

    encoder_output = model.encoder(src_tensor, src_mask)
    # print(encoder_output.shape)
    # print(encoder_output)
    outputs = torch.zeros(max_len).type_as(src_tensor.data)
    outputs[0] = torch.LongTensor([TGT_TOKENIZER.bos_id()])
    # print(outputs.shape, outputs)

    for i in range(1, max_len):
        tgt_mask = create_casual_mask(i)
        tgt_input = outputs[:i].unsqueeze(0)
        print(tgt_input.shape, tgt_input)
        out = model.fc_out(model.decoder(tgt_input, encoder_output, src_mask, tgt_mask))

        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = F.log_softmax(out, dim=-1)
        pred = torch.argmax(prob, dim=-1)
        print(pred)

        # outputs[i] = ix[0][0]
        # if ix[0][0] == TGT_TOKENIZER.eos_id():
        #     break
    print(outputs.tolist())
    return TGT_TOKENIZER.decode_ids(outputs.tolist())



if __name__=='__main__':
    zh = translate("Calling the Big Banks’ Bluff", model=MODEL)
    print(zh)
