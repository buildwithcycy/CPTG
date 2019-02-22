import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import config
from data_utils import START_ID


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_size)
        self.gru = nn.GRU(embedding_size,
                          hidden_size,
                          batch_first=True,
                          num_layers=1,
                          bidirectional=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, seq_len):
        total_length = inputs.size(1)
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded, seq_len, batch_first=True)

        # hidden :[1, b, d]
        outputs, hidden = self.gru(packed)
        encoder_outputs, _ = pad_packed_sequence(outputs,
                                                 batch_first=True,
                                                 total_length=total_length)
        return hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True,
                          bidirectional=False, num_layers=1)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, max_length, init_hidden, att_embedding):
        """

        :param inputs: [b, 1]; START TOKEN
        :param max_length: max length to decode
        :param init_hidden: [1, b, d]
        :param att_embedding: [1, b, d']
        :return:
        """
        embedded = self.embedding(inputs)
        prev_hidden = torch.cat((init_hidden, att_embedding), dim=2)
        logits = []
        sampled_ids = []
        for t in range(max_length):
            _, hidden = self.gru(embedded, prev_hidden)  # [1, b, d]
            logit = self.linear(hidden.squeeze(0))  # [b, |V|]
            logits.append(logit)
            scores = F.softmax(logit, dim=1)
            # hard sampling
            sampled_id = torch.multinomial(scores, num_samples=1)  # [b, 1]
            sampled_ids.append(sampled_id)
            # update inputs
            embedded = self.embedding(sampled_id)
            prev_hidden = hidden

        logits = torch.cat(logits, dim=0)  # [b * t, |V|]
        sampled_ids = torch.cat(sampled_ids, dim=1)  # [b, t]
        return logits, sampled_ids


class Seq2Seq(nn.Module):
    def __init__(self, att_embedding_size, num_atts, ber_prob):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(config.vocab_size,
                               config.embedding_size,
                               config.enc_hidden_size)
        self.decoder = Decoder(config.vocab_size,
                               config.embedding_size,
                               config.dec_hidden_size)

        self.att_embedding = nn.Embedding(num_atts,
                                          att_embedding_size)
        self.sampler = Bernoulli(ber_prob)

    def forward(self, src_inputs, src_len, src_atts, trg_max_len, trg_atts):
        """

        :param src_inputs: [b, t]; x
        :param src_atts : [b, 1]; l
        :param trg_atts: [b, 1]; l
        :return:
        """
        # translation from x to y
        batch_size, src_max_len = list(src_inputs.size(0))

        z_x = self.encoder(src_inputs, src_len)
        l_y = self.att_embedding(trg_atts)
        go_token = torch.LongTensor([[START_ID]] * batch_size)
        logits_y, sampled_ys = self.decoder(go_token, trg_max_len, z_x, l_y)

        # back translation from y to x
        z_y = self.encoder(sampled_ys, )
        gate = self.sampler.sample(sample_shape=z_y.size())
        z_xy = gate * z_x + (1 - gate) * z_y
        l_x = self.att_embedding(src_atts)
        logits_x, sampled_xs = self.decoder(go_token, src_max_len, z_xy, l_x)

        return logits_x  # [b * t, |V|]
