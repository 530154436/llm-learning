#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2025/5/10 14:38
# @function:


def main():
    LOGGER.info("加载Dataset和Tokenizer.")
    train_dataset = MTDataset(config.train_data_path)
    dev_dataset = MTDataset(config.dev_data_path)
    test_dataset = MTDataset(config.test_data_path)
    LOGGER.info("加载DataLoader")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=partial(train_dataset.collate_fn, device=config.device))
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=partial(train_dataset.collate_fn, device=config.device))
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=partial(train_dataset.collate_fn, device=config.device))

    LOGGER.info("初始化模型")
    model = Transformer(config.src_vocab_size, config.tgt_vocab_size,
                        d_model=config.d_model, num_heads=config.n_heads, d_ff=config.d_ff,
                        dropout=config.dropout, N=config.n_layers)
    # print(model)
    LOGGER.info(f'模型训练参数: {count_trainable_parameters(model)}')
    LOGGER.info("训练模型...")
    train(model, train_dataloader, dev_dataloader)


if __name__ == '__main__':
    main()