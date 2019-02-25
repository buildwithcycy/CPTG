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


def build_vocab(file_paths, max_vocab_size=50000):
    counter = dict()
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                words = line.split()
                for word in words:
                    if word not in counter:
                        counter[word] = 1
                    else:
                        counter[word] += 1

    sorted_vocab = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
    sorted_vocab = sorted_vocab[:max_vocab_size]
    word2idx = {word: i for i, (word, freq) in enumerate(sorted_vocab, start=4)}
    word2idx[PAD_TOKEN] = 0
    word2idx[UNK_TOKEN] = 1
    word2idx[START_TOKEN] = 2
    word2idx[STOP_TOKEN] = 3
    idx2word = {i: word for word, i in word2idx.items()}

    return word2idx, idx2word


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
            self.num_total_seqs = len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index]
        label = torch.LongTensor(self.labels[index])
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
        sequence = torch.Tensor(sequence)
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
    labels = torch.cat(labels, dim=0)

    return seqs, seq_lens, labels


def get_loader(file_path, word2idx, is_neg, batch_size=32,
               debug=False, shuffle=True):
    dataset = YelpDataset(file_path, word2idx, is_neg, debug=debug)
    data_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  drop_last=True,
                                  collate_fn=collate_fn)
    return data_loader
