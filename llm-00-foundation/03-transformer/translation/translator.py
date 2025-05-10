#!/usr/bin/env python3
# -*- coding:utf-8 -*--
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from typing import List
from torch.nn.utils.rnn import pad_sequence
from modules.mask import create_padding_mask, create_sequence_mask
from modules.models import Transformer
from translation import config
from translation.sentencepiece_tokenizer import SentencePieceTokenizerWithLang


SRC_TOKENIZER = SentencePieceTokenizerWithLang(lang="en")
TGT_TOKENIZER = SentencePieceTokenizerWithLang(lang="zh")
MODEL = Transformer(src_vocab_size=SRC_TOKENIZER.vocab_size(),
                    tgt_vocab_size=TGT_TOKENIZER.vocab_size(),
                    d_model=config.d_model, num_heads=config.n_heads, d_ff=config.d_ff,
                    dropout=config.dropout, N=config.n_layers)
MODEL.load_state_dict(torch.load(config.model_path,
                                 weights_only=False,
                                 map_location=torch.device(config.device)))
MODEL.eval()


@torch.no_grad()
def translate(sentences: List[str],
              model: Transformer,
              max_len=80):
    """ 翻译
    """
    src_token_ids = []
    for sentence in sentences:
        src_token_ids.append(
            [SRC_TOKENIZER.bos_id()] + SRC_TOKENIZER.encode_as_id(sentence) + [SRC_TOKENIZER.eos_id()]
        )
    # [batch_size, seq_len_src]
    src_tensor = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_token_ids],
                              batch_first=True, padding_value=SRC_TOKENIZER.pad_id())
    batch_size, seq_len_src = src_tensor.size()
    # [batch_size, 1, 1 seq_len_src]
    src_mask = create_padding_mask(src_tensor, pad_token_id=SRC_TOKENIZER.pad_id())
    # [batch_size, seq_len_src, dim_model]
    encoder_output = model.encoder(src_tensor, src_mask)

    # [batch_size, seq_len_tgt], 目标输入序列初始化是以 <bos> 开头，然后预测下一个 token id
    tgt_input = torch.Tensor(batch_size, 1).fill_(TGT_TOKENIZER.bos_id()).type_as(src_tensor.data)
    results = [[] for _ in range(batch_size)]
    stop_flag = [False for _ in range(batch_size)]
    count = 0
    for s in range(1, max_len):
        # 目标输入序列: [batch_size, s]
        # 解码器输入自注意力部分掩码: [batch_size, 1, s, s]
        tgt_mask = create_sequence_mask(tgt_input, pad_token_id=TGT_TOKENIZER.pad_id()).type_as(src_tensor.data)
        # 目标输出序列: [batch_size, s, dim_model]
        tgt_out = model.fc_out(model.decoder(Variable(tgt_input), encoder_output, src_mask, Variable(tgt_mask)))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示、并获取当前位置（输出序列的最后一个位置）最大概率的预测词id
        prob = F.log_softmax(tgt_out[:, -1, :], dim=-1)
        pred = torch.argmax(prob, dim=-1)  # [batch_size]

        # 拼接预测的下一个词 [batch_size, s] => [batch_size, s+1]
        tgt_input = torch.cat((tgt_input, pred.unsqueeze(1)), dim=1)
        for i in range(batch_size):
            if stop_flag[i] is False:
                if pred[i] == TGT_TOKENIZER.eos_id():
                    count += 1
                    stop_flag[i] = True
                else:
                    results[i].append(pred[i].item())
            if count == batch_size:
                break
    return [
        ''.join(TGT_TOKENIZER.decode_ids(tgt_token_ids))
        for tgt_token_ids in results
    ]


if __name__=='__main__':
    _sentences = [
        "Which Way is West for Turkey?",
        "Why didn’t Nokia choose Android earlier?\n"
    ]
    zh = translate(sentences=_sentences, model=MODEL)
    print(zh)
