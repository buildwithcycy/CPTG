import os

import numpy as np
import torch

import config
from data_utils import build_vocab, get_loader
from model import Generator
from train_utils import outputids2words, Trainer


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
        data_loaders = [train_loader, dev_loader]
        trainer = Trainer(embedding, data_loaders)
        trainer.train()
        # train(embedding, train_loader, dev_loader)
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
