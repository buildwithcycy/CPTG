import torch
from data_utils import UNK_ID
import config


def step(model, pos_data, neg_data):
    x_pos, pos_len, l_pos = pos_data
    x_neg, neg_len, l_neg = neg_data
    x_pos = x_pos.to(config.device)
    l_pos = l_pos.to(config.device)
    x_neg = x_neg.to(config.device)
    l_neg = l_neg.to(config.device)
    # forward pass
    pos_logits = model(x_pos, pos_len, l_pos, neg_len, l_neg)
    neg_logits = model(x_neg, neg_len, l_neg, pos_len, l_pos)
    logits = torch.cat((pos_logits, neg_logits), dim=0)
    pos_targets = x_pos.view(-1)
    neg_targets = x_neg.view(-1)
    targets = torch.cat((pos_targets, neg_targets), dim=0)
    return logits, targets


def outputids2words(ids, idx2word):
    words = []
    for id in ids:
        id = id.item()
        if id in idx2word:
            word = idx2word[id]
        else:
            word = idx2word[UNK_ID]
        words.append(word)
    return " ".join(words)
