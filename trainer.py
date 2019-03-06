import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from model import Generator, Discriminator
from train_utils import user_friendly_time, time_since, progress_bar, eta, outputids2words


class Trainer(object):
    def __init__(self, embedding, data_loaders):
        if len(data_loaders) == 2:
            self.train_loader, self.dev_loader = data_loaders
            self.is_train = True
            self.g_optim = None
            self.d_optim = None
        else:
            self.test_loader = data_loaders[0]
            self.is_train = False
        self.vocab_size = embedding.shape[0]
        self.embedding = torch.FloatTensor(embedding).to(config.device)

    def train(self):
        generator = Generator(self.embedding, self.vocab_size, config.att_embedding_size, 2, config.ber_prob)
        generator = generator.to(config.device)
        discriminator = Discriminator(2, config.dec_hidden_size, config.enc_hidden_size)
        discriminator = discriminator.to(config.device)
        self.g_optim = optim.Adam(params=generator.parameters(), lr=config.g_lr)
        self.d_optim = optim.Adam(params=discriminator.parameters(), lr=config.d_lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        num_step = 0
        best_loss = 1e10
        batch_nb = len(self.train_loader)
        for epoch in range(1, config.num_epochs + 1):
            start = time.time()
            for i, train_data in enumerate(self.train_loader):
                batch_idx = i + 1
                num_step += 1
                recon_loss, errG, errD = self.step(generator, discriminator, criterion, train_data)

                generator_loss = recon_loss + errG * config.loss_lambda
                # generator_loss = recon_loss
                msg = "{}/{} {} - ETA : {} - recon : {:.4f}, loss G: {:.4f}, loss D: {:.4f}".format(
                    batch_idx, batch_nb,
                    progress_bar(batch_idx, batch_nb),
                    eta(start, batch_idx, batch_nb),
                    recon_loss, errG, errD)
                print(msg)
            # only see reconstruction loss
            dev_loss = self.evaluate(generator)
            msg = "Epoch {} took {} - final loss : {:.4f} - validation loss : {:.4f}" \
                .format(epoch, user_friendly_time(time_since(start)), generator_loss, dev_loss)
            print(msg)
            if dev_loss < best_loss:
                best_loss = dev_loss
                generator.save(config.save_dir, epoch, num_step)

    def step(self, generator, discriminator, recon_criterion, train_data):
        x, x_len, l_src = train_data
        x = x.to(config.device)
        l_src = l_src.to(config.device)
        l_trg = torch.ones_like(l_src, device=config.device) - l_src
        l_trg = l_trg.to(config.device)

        # forward pass
        recon_logits, hiddens_x, hiddens_y, trg_len = generator(x, x_len, l_src, l_trg)

        # backward pass and update Discriminator
        gan_loss = nn.BCEWithLogitsLoss()
        batch_size = x.size(0)

        # FIXME: detach either hiddens_x or hiddens_y or both?
        self.d_optim.zero_grad()
        real_logits = discriminator(hiddens_x.detach(), x_len, l_src)
        fake_logits_x = discriminator(hiddens_x.detach(), x_len, l_trg)
        fake_logits_y = discriminator(hiddens_y.detach(), trg_len, l_trg, sorting=True)

        real_label = torch.full((batch_size,), 1, device=config.device)
        fake_label = torch.full((batch_size,), 0, device=config.device)
        # 2logD(h_x, l)
        errD_real = 2 * gan_loss(real_logits, real_label)

        # log(1 - D(h_y, l')) + log(1 - D(h_x, l'))
        errD_fake_x = gan_loss(fake_logits_x, fake_label)
        errD_fake_y = gan_loss(fake_logits_y, fake_label)
        errD = errD_real + errD_fake_x + errD_fake_y
        errD.backward()
        self.d_optim.step()

        # backward pass of Generator
        self.g_optim.zero_grad()
        targets = x.view(-1)
        recon_loss = recon_criterion(recon_logits, targets)

        fake_logits_y = discriminator(hiddens_y, trg_len, l_trg, sorting=True)
        errG = 2 * gan_loss(fake_logits_y, real_label)
        generator_loss = recon_loss + config.loss_lambda * errG
        generator_loss.backward()
        self.g_optim.step()

        return recon_loss, errG, errD

    def evaluate(self, generator):
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        losses = []
        for i, eval_data in enumerate(self.dev_loader):
            with torch.no_grad():
                x, x_len, l_src = eval_data
                x = x.to(config.device)
                l_src = l_src.to(config.device)
                l_trg = torch.ones_like(l_src, device=config.device) - l_src
                recon_logits, _, _, _ = generator(x, x_len, l_src, l_trg)
                target = x.view(-1)
                loss = criterion(recon_logits, target)
                losses.append(loss.item())

        return np.mean(losses)

    def inference(self, model_path, output_dir, idx2word):
        model = Generator(self.embedding, self.vocab_size, config.att_embedding_size, 2, config.ber_prob)
        model.load_state_dict(torch.load(model_path))
        model = model.to(config.device)
        total_decoded_sents = []
        original_sents = []

        for test_data in self.test_loader:
            x, x_len, l_src = test_data
            x = x.to(config.device)
            l_trg = torch.zeros_like(l_src, device=config.device)
            decoded = model.decode(x, x_len, l_trg, config.max_decode_step)

            origin_sent = [outputids2words(sent, idx2word) for sent in x.detach()]
            original_sents.extend(origin_sent)

            decoded_sents = [outputids2words(sent, idx2word) for sent in decoded.detach()]
            total_decoded_sents.extend(decoded_sents)

        # write the results into text file
        path = os.path.join(output_dir, "decoded.txt")

        with open(path, "w", encoding="utf-8") as f:
            for original_sent, decoded_sent in zip(original_sents, total_decoded_sents):
                f.write(original_sent + " -> " + decoded_sent + "\n")

        idx = np.random.randint(0, 100)
        decoded_sent = total_decoded_sents[idx]
        original_sent = original_sents[idx]
        print(original_sent, "->", decoded_sent)
