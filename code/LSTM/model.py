import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import os
import sys
import io

class biLSTM(nn.Module):

    def __init__(self, emb_dim, hid_dim, tok_size, label_size, batch_size, pretrained_emb, vocab, dropout=0.5):
        super(biLSTM, self).__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.tok_size = tok_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.pretrained_emb = pretrained_emb
        self.vocab = vocab
        self.dropout = dropout
        if self.pretrained_emb == "rand":
            self.embeddings = nn.Embedding(tok_size, emb_dim)
        else:
            data = {}
            vec_size = -1
            out_of_vocab = np.zeros(0)
            fin = io.open(self.pretrained_emb, 'r', encoding='utf-8', newline='\n', errors='ignore')
            for line in tqdm(fin):
                #print("HEY")
                tokens = line.rstrip().split(' ')
                data[tokens[0]] = [float(v) for v in tokens[1:]] 
                if out_of_vocab.shape[0] == 0:
                    out_of_vocab = np.asarray( data[tokens[0]] )
                if vec_size < 0:
                    vec_size = len(tokens[1:])
            count = len(data)
            #print(count)
            out_of_vocab = np.asarray(out_of_vocab)/count
            emb_mat = np.zeros((len(self.vocab), vec_size))
            for i,j in enumerate(self.vocab):
                emb_mat[i] = data[j]
            self.embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(emb_mat))

        self.lstm = nn.LSTM(self.emb_dim, self.hid_dim, num_layers=1, bidirectional=True)
        self.hiddenToLabel = nn.Linear(self.hid_dim*2,self.label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(2, self.batch_size, self.hid_dim)), Variable(torch.zeros(2, self.batch_size, self.hid_dim)))

    def forward(self, sentence):
        #print(sentence.shape)
        x = self.embeddings(sentence)
        #print("x1", x.shape)
        x = x.view(len(sentence), self.batch_size, -1)
        #print("x2", x.shape)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        #print("lstm_out", lstm_out.shape)
        y = self.hiddenToLabel(lstm_out)
        #print("y", y.shape)
        prob = torch.swapaxes(y,1,2).squeeze(-1)
        prob = F.log_softmax(prob, dim=1)
        #print("prob", prob.shape)
        return prob
