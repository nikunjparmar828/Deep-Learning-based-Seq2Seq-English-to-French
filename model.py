# Nikunj Parmar
# References: 
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#preparing-training-data
# https://github.com/shangeth/Seq2Seq-Machine-Translation

# Glove 6B
# http://nlp.stanford.edu/data/glove.6B.zip

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 25

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        '''
		Encoder:
			- Embedding matrix of size (no of words in corpus, hidden_size) is learned
			- GRU
        '''
        self.hidden_size = hidden_size
        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding = self.embedding_prep()
        self.gru = nn.GRU(hidden_size, hidden_size)
    
    def embedding_prep(self):
        vocab,embeddings = [],[]
        
        with open('glove.6B.50d.txt','rt') as fi:
            full_content = fi.read().strip().split('\n')

        for i in range(len(full_content)):
            i_word = full_content[i].split(' ')[0]
            i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
            vocab.append(i_word)
            embeddings.append(i_embeddings)
        
        vocab_npa = np.array(vocab)
        embs_npa = np.array(embeddings)

        #insert '<pad>' and '<unk>' tokens at start of vocab_npa.
        vocab_npa = np.insert(vocab_npa, 0, '<pad>')
        vocab_npa = np.insert(vocab_npa, 1, '<unk>')

        pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
        unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.

        #insert embeddings for pad and unk tokens at top of embs_npa.
        embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))
        my_embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float())

        return my_embedding_layer

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)