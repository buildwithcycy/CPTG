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
    def __init__(self, file_path, word2idx, debug=False):
        seqs = open(file_path, "r", encoding="utf-8").readlines()
        self.seqs = list(map(lambda line: line.strip(), seqs))
        self.word2idx = word2idx
        self.num_total_seqs = len(self.seqs)
        if debug:
            self.seqs = self.seqs[:100]

    def __getitem__(self, index):
        seq = self.seqs[index]
        seq = self.words2ids(seq, self.word2idx)
        return seq

    def __len__(self):
        return self.num_total_seqs

    def words2ids(self, sentence, word2idx):
        tokens = sentence.lower().split()
        sequence = []
        for token in tokens:
            if token in word2idx:
                sequence.append(word2idx[token])
            else:
                sequence.append(word2idx[UNK_TOKEN])
        sequence.append(word2idx[STOP_TOKEN])
        return sequence


def collate_fn(seqs):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seq = torch.zeros(len(sequences), max(lengths), dtype=torch.long)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seq[i, :end] = seq[:end]
        return padded_seq, lengths

    seqs.sort(key=lambda x: len(x), reverse=True)

    seqs, seq_lens = merge(seqs)

    return seqs, seq_lens


def get_loader(file_path, word2idx, batch_size=32,
               debug=False, shuffle=True):
    dataset = YelpDataset(file_path, word2idx, debug=debug)
    data_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  collate_fn=collate_fn)
    return data_loader
