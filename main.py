import config
from data_utils import build_vocab, get_loader
from trainer import Trainer
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main():
    
    #loading file paths of the sentiment analysis dataset
    train_file_paths = ["data/yelp/sentiment.train.0", "data/yelp/sentiment.train.1"]
    dev_file_paths = ["data/yelp/sentiment.dev.0", "data/yelp/sentiment.dev.1"]
    test_file_paths = ["data/yelp/sentiment.test.0", "data/yelp/sentiment.test.1"]

    print ("I just loaded the filed paths")
    word2idx, idx2word, embedding = build_vocab(train_file_paths,
                                                glove_path=config.glove_path)
    print ("I just loaded the word2ix paths")
    
    if config.train:
        # prepare data loader for training
        print ("I just entered config.train")
        train_loader = get_loader(train_file_paths[1],
                                  train_file_paths[0],
                                  word2idx,
                                  debug=config.debug,
                                  batch_size=config.batch_size)
        # prepare data loader for evaluation
        print ("I just loaded the train_loader")
        dev_loader = get_loader(dev_file_paths[1],
                                dev_file_paths[0],
                                word2idx,
                                shuffle=False,
                                debug=config.debug,
                                batch_size=config.batch_size)
        print ("I just loaded the dev loader paths")
        data_loaders = [train_loader, dev_loader]
        print ("I just loaded the data loaders")
        trainer = Trainer(embedding, data_loaders)
        print ("I just loaded the trainer in main")
        trainer.train()
        print ("I just called the trainer method")
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
