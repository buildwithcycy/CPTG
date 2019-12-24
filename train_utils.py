import time

import numpy as np
import torch

from data_utils import UNK_ID


def get_first_eos_idx(input_ids, eos_id):
    """

    :param input_ids: [b, t]
    :param eos_id: id of EOS token
    :return: [b] the index of first occurence of eos
    """
    mask = input_ids == eos_id
    num_eos = torch.sum(mask, 1)
    # change Tensor to cpu because torch.argmax works differently in cuda and cpu
    # but np.argmax is consistent it returns the first index of the maximum element
    mask = mask.cpu().numpy()
    indices = np.argmax(mask, 1)
    # convert numpy array to Tensor
    seq_len = torch.tensor(indices, dtype=torch.long,
                           device=input_ids.device)

    # in case there is no eos in the sequence
    max_len = input_ids.size(1)
    seq_len = seq_len.masked_fill(num_eos == 0, max_len - 1)

    return seq_len


def outputids2words(ids, idx2word):
    words = []
    for id in ids:
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
