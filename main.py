import config
from data_utils import build_vocab, get_loader
from trainer import Trainer


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
    else:
        test_loader = get_loader(test_file_paths[1],
                                 test_file_paths[0],
                                 word2idx,
                                 debug=config.debug,
                                 shuffle=False,
                                 batch_size=16)
        data_loaders = [test_loader]
        trainer = Trainer(embedding, data_loaders)
        trainer.inference(config.model_path, config.output_dir, idx2word)


if __name__ == "__main__":
    main()
