import time

import numpy as np
import torch

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
    # change Tensor to cpu because torch.argmax works differently in cuda and cpu
    # but np.argmax is more consistent
    mask = mask.cpu().numpy()
    indices = np.argmax(mask, 1)
    # convert numpy array to Tensor
    indices = torch.LongTensor(indices).to(config.device)
    return indices


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
