import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from data_utils import build_vocab, get_loader
from model import Generator, Discriminator
from train_utils import step, outputids2words, eta, progress_bar, user_friendly_time, time_since


def train(embedding, train_loader, dev_loader):
    vocab_size = embedding.shape[0]
    embedding = torch.FloatTensor(embedding).to(config.device)
    generator = Generator(embedding, vocab_size, config.att_embedding_size, 2, config.ber_prob)
    generator = generator.to(config.device)
    discriminator = Discriminator(2, config.dec_hidden_size, config.enc_hidden_size)
    discriminator = discriminator.to(config.device)

    optimizerG = optim.RMSprop(params=generator.parameters(), lr=config.g_lr)
    optimizerD = optim.RMSprop(params=discriminator.parameters(), lr=config.d_lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    num_step = 0
    best_loss = 1e10
    batch_nb = len(train_loader)
    for epoch in range(1, config.num_epochs + 1):
        start = time.time()
        for i, train_data in enumerate(train_loader):
            batch_idx = i + 1
            num_step += 1
            recon_loss, errG, errD = step(generator, discriminator, criterion, train_data)

            # backward pass for discriminator
            optimizerD.zero_grad()
            errD.backward(retain_graph=True)
            optimizerD.step()

            # backward pass for generator
            optimizerG.zero_grad()
            generator_loss = recon_loss + errG * config.loss_lambda
            # generator_loss = recon_loss
            generator_loss.backward()
            optimizerG.step()
            msg = "{}/{} {} - ETA : {} - loss G: {:.4f}, loss D: {:.4f}".format(
                batch_idx, batch_nb,
                progress_bar(batch_idx, batch_nb),
                eta(start, batch_idx, batch_nb),
                generator_loss, errD)
            print(msg, end="\n")

        dev_loss = evaluate(generator, discriminator, dev_loader)
        msg = "Epoch {} took {} - final loss : {:.4f} - validation loss : {:.4f}" \
            .format(epoch, user_friendly_time(time_since(start)), generator_loss, dev_loss)
        print(msg)
        if dev_loss < best_loss:
            best_loss = dev_loss
            generator.save(config.save_dir, epoch, num_step)


def evaluate(generator, discriminator, dev_loader):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    losses = []
    for i, eval_data in enumerate(dev_loader):
        with torch.no_grad():
            recon_loss, errG, errD = step(generator, discriminator, criterion, eval_data)
            loss = recon_loss + errG * config.loss_lambda
            losses.append(loss.item())

    return np.mean(losses)


def inference(model_path, output_dir, data_loader, embedding, idx2word):
    vocab_size = embedding.shape[0]
    embedding = torch.FloatTensor(embedding).to(config.device)
    model = Generator(embedding, vocab_size, config.att_embedding_size, 2, config.ber_prob)
    model.load_state_dict(torch.load(model_path))
    model = model.to(config.device)
    total_decoded_sents = []
    original_sents = []

    for test_data in data_loader:
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


def main():
    train_file_paths = ["data/yelp/sentiment.train.0", "data/yelp/sentiment.train.1"]
    dev_file_paths = ["data/yelp/sentiment.dev.0", "data/yelp/sentiment.dev.1"]
    test_file_paths = ["data/yelp/sentiment.test.0", "data/yelp/sentiment.test.1"]

    word2idx, idx2word, embedding = build_vocab(train_file_paths,
                                                glove_path=config.glove_path)
    if config.train:
        # prepare data loader for training
        train_loader = get_loader(train_file_paths[1],
                                  train_file_paths[0],
                                  word2idx,
                                  debug=config.debug,
                                  batch_size=config.batch_size)
        # prepare data loader for evaluation
        dev_loader = get_loader(dev_file_paths[1],
                                dev_file_paths[0],
                                word2idx,
                                shuffle=False,
                                debug=config.debug,
                                batch_size=config.batch_size)

        train(embedding, train_loader, dev_loader)
    else:
        test_loader = get_loader(test_file_paths[1],
                                 test_file_paths[0],
                                 word2idx,
                                 debug=config.debug,
                                 shuffle=False,
                                 batch_size=16)

        # use train data to decode for debugging
        inference(config.model_path, config.output_dir,
                  test_loader, embedding, idx2word)


if __name__ == "__main__":
    main()
