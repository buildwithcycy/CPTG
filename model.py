import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.nn.utils.rnn import pack_padded_sequence

import config
from data_utils import START_ID, STOP_ID
from train_utils import sequence_mask, make_one_hot, get_first_eos_idx


class Encoder(nn.Module):
    def __init__(self, embedding, vocab_size, embedding_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_size)
        self.embedding = self.embedding.from_pretrained(embedding, freeze=False)
        self.gru = nn.GRU(embedding_size,
                          hidden_size,
                          batch_first=True,
                          num_layers=1,
                          bidirectional=False)

    def forward(self, inputs, seq_len, sorting=False):
        if sorting:
            seq_len, _ = torch.sort(seq_len, descending=True)
            idx_sort = torch.argsort(seq_len, descending=True)
            idx_unsort = torch.argsort(idx_sort)
            inputs = inputs.index_select(0, idx_unsort)
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded, seq_len, batch_first=True)

        # hidden :[1, b, d]
        _, hidden = self.gru(packed)
        if sorting:
            hidden = hidden.index_select(1, idx_unsort)
        return hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True,
                          bidirectional=False, num_layers=1)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, max_length, init_hidden, att_embedding):
        """

        :param inputs: [b, 1]; START TOKEN or [b, t+1] for teacher forcing
        :param max_length: max length to decode
        :param init_hidden: [1, b, d]
        :param att_embedding: [b, d']
        :return:
        """
        batch_size = inputs.size(0)
        total_length = inputs.size(1)
        embedded = self.embedding(inputs)
        att_embedding = att_embedding.unsqueeze(dim=0)

        prev_hidden = torch.cat((init_hidden, att_embedding), dim=2)
        logits = []
        sampled_ids = []
        hiddens = []
        # select only GO_TOKEN embedding when teacher-forcing
        if total_length > 1:
            curr_embedded = embedded[:, 0, :].unsqueeze(1)
        else:
            curr_embedded = embedded

        for t in range(max_length):
            _, hidden = self.gru(curr_embedded, prev_hidden)  # [1, b, d]
            hiddens.append(hidden)
            logit = self.linear(hidden.squeeze(0))  # [b, |V|]
            logits.append(logit)
            scores = F.softmax(logit, dim=1)

            # hard sampling for y
            if total_length == 1:
                sampled_id = torch.multinomial(scores, num_samples=1)  # [b, 1]
                sampled_ids.append(sampled_id)
                # update next token embedding
                curr_embedded = self.embedding(sampled_id)

            else:
                sampled_id = torch.argmax(scores, 1, keepdim=True)
                sampled_ids.append(sampled_id)
                # get next token embedding because it is teacher forcing
                curr_embedded = embedded[:, t + 1, :].unsqueeze(1)
            # update hidden
            prev_hidden = hidden

        hiddens = torch.cat(hiddens, dim=0).transpose(0, 1)  # [t, b, d] -> [b,t,d]
        logits = torch.stack(logits, dim=1)  # [b, t*|V|] why torch.cat(logits, dim=0) does not work?
        logits = logits.view(batch_size * max_length, -1)
        sampled_ids = torch.cat(sampled_ids, dim=1)  # [b, t]

        return hiddens, logits, sampled_ids


class Generator(nn.Module):
    def __init__(self, embedding, vocab_size, att_embedding_size, num_atts, ber_prob):
        super(Generator, self).__init__()
        self.encoder = Encoder(embedding, vocab_size,
                               config.embedding_size,
                               config.enc_hidden_size)
        self.decoder = Decoder(vocab_size,
                               config.embedding_size,
                               config.dec_hidden_size)

        self.decoder.embedding.weight = self.encoder.embedding.weight
        self.att_embedding = nn.Embedding(num_atts,
                                          att_embedding_size)

        self.sampler = Bernoulli(ber_prob)

    def forward(self, src_inputs, src_len, src_attr, trg_attr):
        """

        :param src_inputs: [b, t]; x
        :param src_len: [b] length of valid x
        :param src_attr : [b]; l_x
        :param trg_attr: [b]; l_y
        :return:
        """
        # translation from x to y
        batch_size, src_max_len = list(src_inputs.size())

        z_x = self.encoder(src_inputs, src_len)
        l_y = self.att_embedding(trg_attr)
        go_token = torch.full((batch_size,1), START_ID, dtype=torch.long, device=config.device)

        hiddens_y, logits_y, sampled_ys = self.decoder(go_token,
                                                       config.max_decode_step,
                                                       z_x, l_y)

        # find the idx of the first occurrence EOS
        first_eos_idx = get_first_eos_idx(sampled_ys, STOP_ID)
        trg_len = first_eos_idx + 1

        # mask for hiddens of PAD
        y_mask = sequence_mask(trg_len, config.max_decode_step).unsqueeze(-1)
        hiddens_y = hiddens_y * y_mask

        # back translation from y to x
        z_y = self.encoder(sampled_ys, trg_len, sorting=True)
        # Bernoulli sampling
        gate = self.sampler.sample(sample_shape=z_y.size()).to(config.device)
        z_xy = gate * z_x + (1 - gate) * z_y
        l_x = self.att_embedding(src_attr)

        # for reconstructing x we use teacher forcing
        go_x = torch.cat((go_token, src_inputs), dim=1)
        hiddens_x, recon_logits, sampled_xs = self.decoder(go_x, src_max_len, z_xy, l_x)

        # mask for hiddens of PAD and make them constant
        x_mask = sequence_mask(src_len).unsqueeze(-1)
        hiddens_x = hiddens_x * x_mask

        return recon_logits, hiddens_x, hiddens_y, trg_len  # [b * t, |V|]

    def decode(self, inputs, src_len, trg_attr, max_decode_step):
        """

        :param inputs: source input; [b,t]
        :param src_len: length of valid x [b]
        :param trg_attr: target attribute l_y
        :param max_decode_step: max time steps to decode
        :return:
        """
        with torch.no_grad():
            batch_size = inputs.size(0)
            z_x = self.encoder(inputs, src_len)
            l_y = self.att_embedding(trg_attr)
            start_token = torch.full((batch_size, 1), START_ID,
                                     dtype=torch.long, device=config.device)

            _, _, sampled_ys = self.decoder(start_token, max_decode_step, z_x, l_y)

        return sampled_ys

    def save(self, dir, epoch, step):
        if not os.path.exists(dir):
            os.mkdir(dir)
        model_save_path = os.path.join(dir, str(step) + "_" + str(epoch) + ".ckpt")
        torch.save(self.state_dict(), model_save_path)


class Discriminator(nn.Module):
    def __init__(self, num_labels, dec_hidden_size, hidden_size):
        super(Discriminator, self).__init__()
        self.W = nn.Linear(num_labels, 2 * hidden_size)
        self.v = nn.Linear(2 * hidden_size, 1, bias=False)
        self.gru = nn.RNN(dec_hidden_size, hidden_size,
                          batch_first=True,
                          bidirectional=True)
        self.num_labels = num_labels

    def forward(self, inputs, seq_len, attr_vector, sorting=False):
        """

        :param inputs: [b, t]
        :param seq_len: [b]
        :param attr_vector: [b]
        :param sorting: whether to sort inputs by seq_len
        :return:
        """
        # make one-hot vector
        label_vector = make_one_hot(attr_vector, self.num_labels).to(config.device)

        if sorting:
            seq_len, _ = torch.sort(seq_len, descending=True)
            idx_sort = torch.argsort(seq_len, descending=True)
            idx_unsort = torch.argsort(idx_sort)
            inputs = inputs.index_select(0, idx_sort)

        packed = pack_padded_sequence(inputs, seq_len, batch_first=True)
        _, hidden = self.gru(packed)  # [2, b, d]
        hidden = torch.cat([h for h in hidden], dim=1)  # [b, 2*d]

        if sorting:
            hidden = hidden.index_select(0, idx_unsort)

        l_W = self.W(label_vector)  # [b, 2*d]
        l_W_phi = torch.sum(l_W * hidden, dim=1, keepdim=True)  # [b,1]
        v_phi = self.v(hidden)  # [b, 1]
        prob = torch.sigmoid(l_W_phi + v_phi).squeeze(1)  # [b]

        return prob
