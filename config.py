import torch

batch_size = 64
num_epochs = 50
vocab_size = 50000
embedding_size = 300
enc_hidden_size = 500
dec_hidden_size = 700
att_embedding_size = 200
ber_prob = 0.5
d_lr = 1e-5
g_lr = 1e-3
loss_lambda = 0.1
max_decode_step = 20

train = True
debug = True
save_dir = "./save"
model_path = "./save/3.ckpt"
output_dir = "./result"
device = torch.device("cuda:1")
