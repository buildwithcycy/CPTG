import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from data_utils import build_vocab, get_loader
from model import Seq2Seq
from train_utils import step, outputids2words


def train(model, train_data, dev_data):
    train_pos_loader, train_neg_loader = zip(*train_data)
    dev_pos_loader, dev_neg_loader = zip(*dev_data)

    optimizer = optim.Adam(params=model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    num_step = 0
    best_loss = 1e10
    for epoch in range(1, config.num_epochs + 1):
        for i, pos_data in enumerate(train_pos_loader):
            for j, neg_data in enumerate(train_neg_loader):
                num_step += 1
                logits, targets = step(model, pos_data, neg_data)
                loss = criterion(logits, targets)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                print(epoch, num_step, loss.item())
        dev_loss = evaluate(model, dev_pos_loader, dev_neg_loader)
        if dev_loss < best_loss:
            best_loss = dev_loss
            print("new score at {}, loss : {}".format(epoch, dev_loss))
            model.save(config.save_dir, epoch, num_step)


def evaluate(model, pos_data_loader, neg_data_loader):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    losses = []
    for i, pos_data in enumerate(pos_data_loader):
        for j, neg_data in enumerate(neg_data_loader):
            with torch.no_grad():
                logits, targets = step(model, pos_data, neg_data)
                loss = criterion(logits, targets)
                losses.append(loss.item())

    return np.mean(losses)


def inference(model, pos_data_loader, neg_data_loader, idx2word):
    total_pos_sents = []
    total_neg_sents = []
    original_pos_sents = []
    original_neg_sents = []
    for pos_data in pos_data_loader:
        for neg_data in neg_data_loader:
            x_pos, pos_len, l_pos = pos_data
            x_neg, neg_len, l_neg = neg_data

            x_pos = x_pos.to(config.device)
            l_pos = l_pos.to(config.device)
            x_neg = x_neg.to(config.device)
            l_neg = l_neg.to(config.device)

            pos2neg = model.decode(x_pos, pos_len, l_neg, config.max_decode_step)
            neg2pos = model.decode(x_neg, neg_len, l_pos, config.max_decode_step)

            origin_pos_sent = [outputids2words(x, idx2word) for x in x_pos.detach()]
            origin_neg_sent = [outputids2words(x, idx2word) for x in x_neg.detach()]
            original_pos_sents.extend(origin_pos_sent)
            original_neg_sents.extend(origin_neg_sent)

            neg_sents = [outputids2words(x, idx2word) for x in pos2neg.detach()]
            pos_sents = [outputids2words(x, idx2word) for x in neg2pos.detach()]
            total_neg_sents.extend(neg_sents)
            total_pos_sents.extend(pos_sents)

    # write the results into text file
    with open("result/pos2neg.txt", "w", encoding="utf-8") as f:
        for original_pos_sent, decoded_pos_sent in zip(original_pos_sents, total_pos_sents):
            f.write(original_pos_sent + " -> " + decoded_pos_sent + "\n")
    with open("result/neg2pos.txt", "w", encoding="utf-8") as f:
        for original_neg_sent, decoded_neg_sent in zip(original_neg_sents, total_neg_sents):
            f.write(original_neg_sent + " -> " + decoded_neg_sent + "\n")

    return total_pos_sents, total_neg_sents, original_pos_sents, original_neg_sents


def main():
    train_file_paths = ["data/yelp/sentiment.train.0", "data/yelp/sentiment.train.1"]
    dev_file_paths = ["data/yelp/sentiment.dev.0", "data/yelp/sentiment.dev.1"]
    word2idx, idx2word = build_vocab(train_file_paths)
    model = Seq2Seq(config.att_embedding_size, 2, config.ber_prob)
    model.to(config.device)
    # prepare data loader for training
    train_pos_loader = get_loader(train_file_paths[1], word2idx,
                                  is_neg=False,
                                  debug=config.debug,
                                  shuffle=True,
                                  batch_size=16)
    train_neg_loader = get_loader(train_file_paths[0], word2idx,
                                  is_neg=True,
                                  shuffle=True,
                                  debug=config.debug,
                                  batch_size=16)
    # prepare data loader for evaluation
    dev_pos_loader = get_loader(dev_file_paths[1],
                                word2idx,
                                is_neg=False,
                                debug=config.debug,
                                batch_size=16)
    dev_neg_loader = get_loader(dev_file_paths[0],
                                word2idx,
                                is_neg=True,
                                batch_size=16)

    # for debugging we use train-set
    test_pos_loader = get_loader(train_file_paths[1], word2idx,
                                 is_neg=False,
                                 debug=config.debug,
                                 shuffle=False,
                                 batch_size=16)
    test_neg_loader = get_loader(train_file_paths[0], word2idx,
                                 is_neg=True,
                                 shuffle=False,
                                 debug=config.debug,
                                 batch_size=16)

    train_data = zip(train_pos_loader, train_neg_loader)
    dev_data = zip(dev_pos_loader, dev_neg_loader)
    train(model, train_data, dev_data)
    # use train data to decode for debugging
    pos_sents, neg_sents, original_pos_sents, original_neg_sents = inference(model, test_pos_loader,
                                                                             test_neg_loader, idx2word)
    idx = np.random.randint(0, 100)
    pos_sent = pos_sents[idx]
    neg_sent = neg_sents[idx]
    original_pos_sent = original_pos_sents[idx]
    original_neg_sent = original_neg_sents[idx]
    print(original_pos_sent, "->", neg_sent)
    print(original_neg_sent, "->", pos_sent)


if __name__ == "__main__":
    main()
