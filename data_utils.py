import numpy as np
import torch
import torch.utils.data as data

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
START_TOKEN = "<BOS>"
STOP_TOKEN = "<EOS>"

PAD_ID = 0
UNK_ID = 1
START_ID = 2
STOP_ID = 3


class YelpDataset(data.Dataset):
    def __init__(self, file_path, word2idx, is_neg, debug=False):
        seqs = open(file_path, "r", encoding="utf-8").readlines()
        self.seqs = list(map(lambda line: line.strip(), seqs))
        self.word2idx = word2idx
        self.num_total_seqs = len(self.seqs)
        if is_neg:
            self.labels = np.zeros((self.num_total_seqs, 1))
        else:
            self.labels = np.ones((self.num_total_seqs, 1))
        if debug:
            self.seqs = self.seqs[:100]
            self.labels = self.labels[:100]

    def __getitem__(self, index):
        seq = self.seqs[index]
        label = self.labels[index]
        seq = self.words2ids(seq)
        return seq, label

    def __len__(self):
        return self.num_total_seqs

    def words2ids(self, sentence):
        tokens = sentence.lower().split()
        sequence = []
        for token in tokens:
            if token in self.word2idx:
                sequence.append(self.word2idx[token])
            else:
                sequence.append(self.word2idx[UNK_TOKEN])
        sequence.append(self.word2idx[STOP_TOKEN])
        return sequence


def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seq = torch.zeros(len(sequences), max(lengths), dtype=torch.long)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seq[i, :end] = seq[:end]
        return padded_seq, lengths

    data.sort(key=lambda x: len(x[0]), reverse=True)

    seqs, labels = zip(*data)
    seqs, seq_lens = merge(seqs)

    return seqs, seq_lens


def get_loader(file_path, word2idx, is_neg, batch_size=32,
               debug=False, shuffle=True):
    dataset = YelpDataset(file_path, word2idx, is_neg, debug=debug)
    data_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  collate_fn=collate_fn)
    return data_loader
