import time

import torch
import torch.nn as nn

import config
from data_utils import UNK_ID


def sequence_mask(sequence_length, max_len=None):
    # from https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/array_ops.py
    # The basic idea is to compare a range row vector of size maxlen:
    # [0, 1, 2, 3, 4]
    # to length as a matrix with 1 column: [[1], [3], [2]].
    # Because of broadcasting on both arguments this comparison results
    # in a matrix of size (len(lengths), maxlen)
    if max_len is None:
        max_len = max(sequence_length)

    row_vector = torch.arange(0, max_len).to(config.device)
    matrix = torch.cuda.LongTensor(sequence_length).unsqueeze(-1)
    result = row_vector < matrix
    result = result.float().to(config.device)
    return result


def get_first_eos_idx(inputs, eos_id):
    """

    :param inputs: [b, t]
    :param eos_id: id of EOS token
    :return: [b] the first index of EOS token in inputs
    """
    mask = inputs == eos_id
    indices = torch.argsort(mask, dim=1)[:, 0]

    return indices


def step(generator, discriminator, recon_criterion, train_data):
    x, x_len, l_src = train_data
    x = x.to(config.device)
    l_src = l_src.to(config.device)
    l_trg = torch.ones_like(l_src, device=config.device) - l_src
    l_trg = l_trg.to(config.device)

    # forward pass
    # reconstruction loss
    recon_logits, hiddens_x, hiddens_y, trg_len = generator(x, x_len, l_src, l_trg)

    targets = x.view(-1)
    recon_loss = recon_criterion(recon_logits, targets)
    # adversarial loss
    real_logits = discriminator(hiddens_x, x_len, l_src)
    fake_logits_x = discriminator(hiddens_x, x_len, l_trg)
    fake_logits_y = discriminator(hiddens_y, trg_len, l_trg, sorting=True)

    errD, errG = get_adv_loss(real_logits,
                              fake_logits_x,
                              fake_logits_y)

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
    label = torch.full((batch_size,), fake_label, device=config.device)
    errD_fake_y = criterion(fake_logits_y, label)
    errD_fake_x = criterion(fake_logits_x, label)
    errD_fake = errD_fake_x + errD_fake_y

    errD = errD_real + errD_fake

    # loss for generator
    label = torch.full((batch_size,), real_label, device=config.device)
    errG = criterion(fake_logits_y, label)

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


def make_one_hot(attr, num_labels):
    batch_size = attr.size(0)
    one_hot = torch.zeros(batch_size, num_labels)
    one_hot[range(batch_size), attr] = 1
    return one_hot


def time_since(t):
    """ Function for time. """
    return time.time() - t


def progress_bar(completed, total, step=5):
    """ Function returning a string progress bar. """
    percent = int((completed / total) * 100)
    bar = '[='
    arrow_reached = False
    for t in range(step, 101, step):
        if arrow_reached:
            bar += ' '
        else:
            if percent // t != 0:
                bar += '='
            else:
                bar = bar[:-1]
                bar += '>'
                arrow_reached = True
    if percent == 100:
        bar = bar[:-1]
        bar += '='
    bar += ']'
    return bar


def user_friendly_time(s):
    """ Display a user friendly time from number of second. """
    s = int(s)
    if s < 60:
        return "{}s".format(s)

    m = s // 60
    s = s % 60
    if m < 60:
        return "{}m {}s".format(m, s)

    h = m // 60
    m = m % 60
    if h < 24:
        return "{}h {}m {}s".format(h, m, s)

    d = h // 24
    h = h % 24
    return "{}d {}h {}m {}s".format(d, h, m, s)


def eta(start, completed, total):
    """ Function returning an ETA. """
    # Computation
    took = time_since(start)
    time_per_step = took / completed
    remaining_steps = total - completed
    remaining_time = time_per_step * remaining_steps

    return user_friendly_time(remaining_time)
