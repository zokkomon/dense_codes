import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from collections import Counter
from torchtext.vocab import vocab
from torchtext.data import get_tokenizer
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint


# To install spacy languages do:
# python -m spacy download en
# python -m spacy download de


spacy_german = spacy.load("de_core_news_sm")
spacy_english = spacy.load("en_core_news_sm")

def tokenizer_german(text):
    return [tok.text for tok in spacy_german.tokenizer(text)]

def tokenizer_english(text):
    return [tok.text for tok in spacy_english.tokenizer(text)]

# german = Field(tokenize=tokenizer_german, lower=True, init_token="sos", eos_token="eos")
# english = Field(tokenize=tokenizer_english, lower=True, init_token="sos", eos_token="eos")

# train_data, valid_data, test_data = Multi30k.spilts(
#     exts=(".de",".en"), Field=(german,english))

# german.build_vocab(train_data, max_size=10000, min_freq=2)
# english.build_vocab(train_data, max_size=10000, min_freq=2)

class Transformer(nn.Module):
    def __init__(
            self,
            embedding_size,
            src_vocab_size,
            tgt_vocab_size,
            src_pad_idx,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
            max_len
            ):
        super(Transformer,self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size,embedding_size)
        self.src_position_embedding = nn.Embedding(max_len,embedding_size)
        self.tgt_word_embedding = nn.Embedding(tgt_vocab_size,embedding_size)
        self.tgt_position_embedding = nn.Embedding(max_len,embedding_size)
        
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers, 
            forward_expansion,
            dropout,
            custom_encoder=None,
            custom_decoder=None,
            layer_norm_eps=1e-05,
            batch_first=False,
            norm_first=False
            )
        
        self.dense = nn.Linear(embedding_size,tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx
        
    def make_src_mask(self,src):
        # src shape: src_len,N
        src_mask = src.transpose(0,1) == self.src_pad_idx
        #(N,src_len)
        return src_mask
    
    def forward(self, src, tgt):
        src_seq_len, N = src.shape
        tgt_seq_len, N = tgt.shape
        
        src_position = (
            torch.arange(0,src_seq_len).unqueeze(1).expand(src_seq_len, N)
            )
        
        tgt_position = (
            torch.arange(0,tgt_seq_len).unqueeze(1).expand(tgt_seq_len, N)
            )
        
        embed_src = self.dropout(
            self.src_word_embedding(src) + self.src_position_embedding(src_position)
            )
        
        embed_tgt = self.dropout(
            self.tgt_word_embedding(tgt) + self.tgt_position_embedding(tgt_position)
            )
        
        src_mask = self.make_src_mask(src)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len)
        
        x = self.transformer(
            embed_src,
            embed_tgt,
            src_key_padding_mask = src_mask,
            tgt_mask = tgt_mask
            )
        x = self.dense(x)
        
        return x
    
# Setup and Training
device = torch.device("cpu")
load_model = False
save_model = True

# Training Hyperparameter
num_epochs = 5
lr = 3e-4
batch_size = 32

# MOdel Hyperparameter
src_vocab_size = len(german.vocab)
tgt_vocab_size = len(english.vocab)
embedding_size = 512
num_heads = 8
n_encoder = 3
n_decoder = 3
forward_expansion = 4
dropout = 0.1
max_len = 100
src_pad_idx = english.vocab.stoi("<pad>")

# Tensorboard 
writer = SummaryWriter("runs/loss_plot")
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.spilts(
    (train_data,valid_data,test_data),
    batch_size = batch_size,
    sort_within_batch = True,
    sort_key = lambda x: len(x.src),
    device = device
    )

translation_model = Transformer(
    embedding_size,
    src_vocab_size,
    tgt_vocab_size,
    src_pad_idx,
    num_heads,
    n_encoder,
    n_decoder,
    forward_expansion,
    dropout,
    max_len
    ).to(device)

optimizer = optim.Adam(translation_model.paramaters(), lr=lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

pad_idx = english.vocab.stoi("<pad>")
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.ptar"), translation_model, optimizer)
    
sentence = "ein pferd geht unter einer brucke neben einem boot"

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")
    
    if save_model:
        checkpoint = {
            "state_dict" : translation_model.state_dict(),
            "optimizer" : optimizer.state_dict()
            }
        save_checkpoint(checkpoint)
        
    translation_model.eval()
    translated_sentence = translate_sentence(
        translation_model, sentence, german, english, device, max_length = 100
        )
    
    print(f"Translated sentence \n {translated_sentence}")
    translation_model.train()
    losses = []
    
    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.tgt.to(device)

        # Forward prop
        output = translation_model(inp_data, target[:-1, :])

        # Output is of shape (tgt_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin.
        # Let's also remove the start token while we're at it
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())

        # Back prop
        loss.backward()
        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(translation_model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)

# running on entire test data takes a while
score = bleu(test_data[1:100], translation_model, german, english, device)
print(f"Bleu score {score * 100:.2f}")    
        