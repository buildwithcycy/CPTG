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


def train(embedding, train_data, dev_data):
    embedding = torch.FloatTensor(embedding).to(config.device)
    vocab_size = embedding.shape[0]
    generator = Generator(embedding, vocab_size, config.att_embedding_size, 2, config.ber_prob)
    generator = generator.to(config.device)
    discriminator = Discriminator(2, config.dec_hidden_size, config.enc_hidden_size)
    discriminator = discriminator.to(config.device)

    train_pos_loader, train_neg_loader = zip(*train_data)
    dev_pos_loader, dev_neg_loader = zip(*dev_data)

    optimizerG = optim.RMSprop(params=generator.parameters(), lr=config.g_lr)
    optimizerD = optim.RMSprop(params=discriminator.parameters(), lr=config.d_lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    num_step = 0
    best_loss = 1e10
    batch_nb = len(train_pos_loader)
    for epoch in range(1, config.num_epochs + 1):
        start = time.time()
        for i, train_data in enumerate(zip(train_pos_loader, train_neg_loader)):
            pos_data, neg_data = train_data
            batch_idx = i + 1
            num_step += 1
            recon_loss, errG, errD = step(generator, discriminator, criterion, pos_data, neg_data)

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

        dev_loss = evaluate(generator, discriminator, dev_pos_loader, dev_neg_loader)
        msg = "Epoch {} took {} - final loss : {:.4f} - validation loss : {:.4f}" \
            .format(epoch, user_friendly_time(time_since(start)), generator_loss, dev_loss)
        print(msg)
        if dev_loss < best_loss:
            best_loss = dev_loss
            generator.save(config.save_dir, epoch, num_step)


def evaluate(generator, discriminator, pos_data_loader, neg_data_loader):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    losses = []
    for i, eval_data in enumerate(zip(pos_data_loader, neg_data_loader)):
        pos_data, neg_data = eval_data
        with torch.no_grad():
            recon_loss, errG, errD = step(generator, discriminator, criterion, pos_data, neg_data)
            # loss = recon_loss + errG * config.loss_lambda
            loss = recon_loss
            losses.append(loss.item())

    return np.mean(losses)


def inference(model_path, output_dir, pos_data_loader, neg_data_loader, idx2word):
    model = Generator(config.att_embedding_size, 2, config.ber_prob)
    model.load_state_dict(model_path)
    total_decoded_pos_sents = []
    total_decoded_neg_sents = []
    original_pos_sents = []
    original_neg_sents = []

    for pos_data in pos_data_loader:
        x_pos, pos_len, l_pos = pos_data
        x_pos = x_pos.to(config.device)
        l_neg = torch.zeros_like(l_pos, device=config.device)
        pos2neg = model.decode(x_pos, pos_len, l_neg, config.max_decode_step)

        origin_pos_sent = [outputids2words(x, idx2word) for x in x_pos.detach()]
        original_pos_sents.extend(origin_pos_sent)

        neg_sents = [outputids2words(x, idx2word) for x in pos2neg.detach()]
        total_decoded_neg_sents.extend(neg_sents)

    for neg_data in neg_data_loader:
        x_neg, neg_len, l_neg = neg_data
        x_neg = x_neg.to(config.device)
        l_pos = torch.ones_like(l_neg, device=config.device)

        neg2pos = model.decode(x_neg, neg_len, l_pos, config.max_decode_step)

        origin_neg_sent = [outputids2words(x, idx2word) for x in x_neg.detach()]
        original_neg_sents.extend(origin_neg_sent)

        pos_sents = [outputids2words(x, idx2word) for x in neg2pos.detach()]

        total_decoded_pos_sents.extend(pos_sents)

    # write the results into text file
    pos2neg_path = os.path.join(output_dir, "pos2neg.txt")
    neg2pos_path = os.path.join(output_dir, "neg2pos.txt")

    with open(pos2neg_path, "w", encoding="utf-8") as f:
        for original_pos_sent, decoded_pos_sent in zip(original_pos_sents, total_decoded_neg_sents):
            f.write(original_pos_sent + " -> " + decoded_pos_sent + "\n")

    with open(neg2pos_path, "w", encoding="utf-8") as f:
        for original_neg_sent, decoded_neg_sent in zip(original_neg_sents, total_decoded_pos_sents):
            f.write(original_neg_sent + " -> " + decoded_neg_sent + "\n")

    idx = np.random.randint(0, 100)
    pos_sent = total_decoded_pos_sents[idx]
    neg_sent = total_decoded_neg_sents[idx]
    original_pos_sent = original_pos_sents[idx]
    original_neg_sent = original_neg_sents[idx]
    print(original_pos_sent, "->", neg_sent)
    print(original_neg_sent, "->", pos_sent)


def main():
    train_file_paths = ["data/yelp/sentiment.train.0", "data/yelp/sentiment.train.1"]
    dev_file_paths = ["data/yelp/sentiment.dev.0", "data/yelp/sentiment.dev.1"]
    test_file_paths = ["data/yelp/sentiment.test.0", "data/yelp/sentiment.test.1"]

    word2idx, idx2word, embedding = build_vocab(train_file_paths,
                                                glove_path=config.glove_path)
    print(embedding.shape)
    if config.train:
        # prepare data loader for training
        train_pos_loader = get_loader(train_file_paths[1], word2idx,
                                      is_neg=False,
                                      debug=config.debug,
                                      batch_size=config.batch_size,
                                      drop_last=True)
        train_neg_loader = get_loader(train_file_paths[0], word2idx,
                                      is_neg=True,
                                      debug=config.debug,
                                      batch_size=config.batch_size,
                                      drop_last=True)
        # prepare data loader for evaluation
        dev_pos_loader = get_loader(dev_file_paths[1],
                                    word2idx,
                                    is_neg=False,
                                    shuffle=False,
                                    debug=config.debug,
                                    batch_size=config.batch_size,
                                    drop_last=True)
        dev_neg_loader = get_loader(dev_file_paths[0],
                                    word2idx,
                                    shuffle=False,
                                    is_neg=True,
                                    batch_size=config.batch_size,
                                    drop_last=True)

        train_data = zip(train_pos_loader, train_neg_loader)
        dev_data = zip(dev_pos_loader, dev_neg_loader)
        train(embedding, train_data, dev_data)
    else:
        test_pos_loader = get_loader(test_file_paths[1], word2idx,
                                     is_neg=False,
                                     debug=config.debug,
                                     shuffle=False,
                                     batch_size=16)
        test_neg_loader = get_loader(test_file_paths[0], word2idx,
                                     is_neg=True,
                                     shuffle=False,
                                     debug=config.debug,
                                     batch_size=16)

        # use train data to decode for debugging
        inference(config.model_path, config.output_dir,
                  test_pos_loader, test_neg_loader, idx2word)


if __name__ == "__main__":
    main()
