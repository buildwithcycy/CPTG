import numpy as np
import torch
import torch.utils.data as data
import config
from tqdm import tqdm

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "UNKNOWN"
START_TOKEN = "<s>"
STOP_TOKEN = "EOS"

PAD_ID = 0
UNK_ID = 1
START_ID = 2
STOP_ID = 3


def build_vocab(file_paths, max_vocab_size=50000, glove_path=None):
    counter = dict()
    word2embedding = dict()
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

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=int(2.2e6)):
            if config.debug:
                break
            word_vec = line.split(" ")
            word = word_vec[0]
            vec = np.array(word_vec[1:], dtype=np.float32)
            word2embedding[word] = vec

    embedding = np.zeros((len(word2idx), 300), dtype=np.float32)
    for word, vec in word2embedding.items():
        if config.debug:
            break
        try:
            idx = word2idx[word]
            embedding[idx] = vec
        except KeyError:
            continue

    return word2idx, idx2word, embedding


class YelpDataset(data.Dataset):
    def __init__(self, pos_file_path, neg_file_path, word2idx, debug=False):
        pos_seqs = open(pos_file_path, "r", encoding="utf-8").readlines()
        neg_seqs = open(neg_file_path, "r", encoding="utf-8").readlines()
        pos_seqs = list(map(lambda line: line.strip(), pos_seqs))
        neg_seqs = list(map(lambda line: line.strip(), neg_seqs))

        pos_labels = np.ones((len(pos_seqs), 1))
        neg_labels = np.zeros((len(neg_seqs), 1))
        self.seqs = pos_seqs + neg_seqs
        self.labels = np.concatenate([pos_labels, neg_labels], axis=0)
        self.num_total_seqs = len(self.seqs)
        self.word2idx = word2idx
        if debug:
            self.seqs = self.seqs[:100]
            self.labels = self.labels[:100]
            self.num_total_seqs = len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index]
        label = torch.tensor(self.labels[index],
                             dtype=torch.long)
        seq = self.words2ids(seq)
        return seq, label

    def __len__(self):
        return self.num_total_seqs

    def words2ids(self, sentence):
        tokens = sentence.lower().split()
        sequence = list()
        sequence.append(self.word2idx[START_TOKEN])
        for token in tokens:
            if token in self.word2idx:
                sequence.append(self.word2idx[token])
            else:
                sequence.append(self.word2idx[UNK_TOKEN])
        sequence.append(self.word2idx[STOP_TOKEN])
        sequence = torch.tensor(sequence, dtype=torch.long)
        return sequence


def collate_fn(data):
    def merge(sequences):
        lengths = torch.tensor([len(seq) for seq in sequences],
                               dtype=torch.long)
        padded_seq = torch.zeros(len(sequences), max(lengths), dtype=torch.long)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seq[i, :end] = seq[:end]
        return padded_seq, lengths

    seqs, labels = zip(*data)
    seqs, seq_lens = merge(seqs)
    labels = torch.cat(labels, dim=0)

    return seqs, seq_lens, labels


def get_loader(pos_file_path, neg_file_path,
               word2idx, batch_size=32,
               debug=False, shuffle=True):

    dataset = YelpDataset(pos_file_path, neg_file_path, word2idx, debug=debug)
    data_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  collate_fn=collate_fn)
    return data_loader
