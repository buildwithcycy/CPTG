import torch

num_epochs = 25
vocab_size = 50000
embedding_size = 300
enc_hidden_size = 500
dec_hidden_size = 700
att_embedding_size = 200
ber_prob = 0.5
lr = 1e-3
loss_lambda = 0.5
max_decode_step = 15
debug = True
save_dir = "./save"
device = torch.device("cuda:3")
