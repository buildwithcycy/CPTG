import torch
import torch.nn as nn
from data_utils import UNK_ID
import config


def sequence_mask(sequence_length):
    # from https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/array_ops.py
    # The basic idea is to compare a range row vector of size maxlen:
    # [0, 1, 2, 3, 4]
    # to length as a matrix with 1 column: [[1], [3], [2]].
    # Because of broadcasting on both arguments this comparison results
    # in a matrix of size (len(lengths), maxlen)
    row_vector = torch.arange(0, max(sequence_length))
    matrix = torch.LongTensor(sequence_length).unsqueeze(-1)
    result = row_vector < matrix
    result = result.float().to(config.device)
    return result


def step(generator, discriminator, criterion, pos_data, neg_data):
    x_pos, pos_len, l_pos = pos_data
    x_neg, neg_len, l_neg = neg_data
    x_pos = x_pos.to(config.device)
    l_pos = l_pos.to(config.device)
    x_neg = x_neg.to(config.device)
    l_neg = l_neg.to(config.device)
    # forward pass
    # reconstruction loss
    recon_pos_logits, pos_hiddens_x, pos_hiddens_y = generator(x_pos, pos_len, l_pos, neg_len, l_neg)
    recon_neg_logits, neg_hiddens_x, neg_hiddens_y = generator(x_neg, neg_len, l_neg, pos_len, l_pos)
    recon_logits = torch.cat((recon_pos_logits, recon_neg_logits), dim=0)
    pos_targets = x_pos.view(-1)
    neg_targets = x_neg.view(-1)
    recon_targets = torch.cat((pos_targets, neg_targets), dim=0)
    recon_loss = criterion(recon_logits, recon_targets)
    # adversarial loss
    pos_real_logits = discriminator(pos_hiddens_x, pos_len, l_pos)
    pos_fake_logits_x = discriminator(pos_hiddens_x, pos_len, l_neg)
    pos_fake_logits_y = discriminator(pos_hiddens_y, neg_len, l_pos)

    pos_errD, pos_errG = get_adv_loss(pos_real_logits,
                                      pos_fake_logits_x,
                                      pos_fake_logits_y)

    neg_real_logits = discriminator(neg_hiddens_x, neg_len, l_neg)
    neg_fake_logits_x = discriminator(neg_hiddens_x, neg_len, l_pos)
    neg_fake_logits_y = discriminator(neg_hiddens_y, pos_len, l_neg)

    neg_errD, neg_errG = get_adv_loss(neg_real_logits,
                                      neg_fake_logits_x,
                                      neg_fake_logits_y)
    errD = (pos_errD + neg_errD) / 2
    errG = (neg_errG + neg_errG) / 2

    return recon_loss, errG, errD


def get_adv_loss(real_logits, fake_logits_x, fake_logits_y):
    criterion = nn.BCELoss()
    real_label = 1
    fake_label = 0
    batch_size = real_logits.size(0)
    label = torch.full((batch_size,), real_label, device=config.device)
    # loss for discriminator
    errD_real = 2 * criterion(real_logits, label)  # 2logD(h_x, l)

    # log(1 - D(h_y, l')) + log(1 - D(h_x, l'))
    label.fill_(fake_label)
    errD_fake_y = criterion(fake_logits_y, label)
    errD_fake_x = criterion(fake_logits_x, label)
    errD_fake = (errD_fake_x + errD_fake_y) / 2

    errD = errD_real + errD_fake

    # loss for generator
    label.fill_(real_label)
    errG_y = criterion(fake_logits_y, label)
    errG_x = criterion(fake_logits_x, label)
    errG = (errG_y + errG_x) / 2

    return errD, errG


def outputids2words(ids, idx2word):
    words = []
    for id in ids:
        id = id.item()
        if id in idx2word:
            word = idx2word[id]
        else:
            word = idx2word[UNK_ID]

        words.append(word)
    try:
        fst_eos_idx = words.index("<EOS>")
        words = words[:fst_eos_idx]
    except ValueError:
        words = words
    sentence = " ".join(words)
    return sentence
