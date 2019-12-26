# Content Preserving Text Generation with Attribute Controls
Pytorch implementation of [Content Preserving Text Generation with Attribute Controls](https://arxiv.org/abs/1811.01135) (Logeswaran et al., 2018) 

## Dataset
* You can download Yelp dataset from [here](https://github.com/shentianxiao/language-style-transfer) 
```bash
git clone https://github.com/shentianxiao/language-style-transfer.git
mv data/yelp ./
rm -rf language-style-transfer
```
* You should download [GLoVE](https://nlp.stanford.edu/pubs/glove.pdf)
```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O ./data/glove.840B.300d.zip 
unzip ./data/glove.840B.300d.zip
```

## How to use
* Set configuration in config.py. 
* If you want to train the model with gpu, set train=True and device = "device:0".
* If you want to test the model, set train=False and model_path = "your model checkpoint" 

## Dependencies
* python >= 3.6
* pytorch >= 1.1
* tqdm


