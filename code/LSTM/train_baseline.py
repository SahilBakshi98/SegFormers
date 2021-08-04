import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time, random
import os, sys
from tqdm import tqdm
import numpy as np
from model import biLSTM
import argparse
from sklearn import metrics

torch.set_num_threads(6)
torch.manual_seed(1)

data_folder = "../../data"
fasttext_folder = "./embeddings"
   
def inp_data(data_name):
    dev_t = os.path.join(data_folder, data_name, data_name + '_dev.tok')
    test_t = os.path.join(data_folder, data_name, data_name + '_test.tok')
    train_t = os.path.join(data_folder, data_name, data_name + '_train.tok')
    dev_c = os.path.join(data_folder, data_name, data_name + '_dev.conllu')
    test_c = os.path.join(data_folder, data_name, data_name + '_test.conllu')
    train_c = os.path.join(data_folder, data_name, data_name + '_train.conllu')
    
    if os.path.isfile(dev_t) == False or os.path.isfile(test_t) == False or os.path.isfile(train_t) == False or os.path.isfile(dev_c) == False or os.path.isfile(test_c) == False or os.path.isfile(train_c) == False:
        print ("File does not exist", file=sys.stderr)

    tok_dev_data = prep_tok_data(dev_t)
    tok_test_data = prep_tok_data(test_t)
    tok_train_data = prep_tok_data(train_t)
    conllu_dev_data = prep_conll_data(dev_c)
    conllu_test_data = prep_conll_data(test_c)
    conllu_train_data = prep_conll_data(train_c)
    
    tok_vocabulary = vocab(tok_train_data)
    conllu_vocabulary = vocab(conllu_train_data)

    if "pdtb" in data_name:
        tagset = {'_':0, 'Seg=B-Conn':1, 'Seg=I-Conn':2}
    else:
        tagset = {'_':0, 'BeginSeg=Yes':1}
    
    return tok_dev_data, tok_test_data, tok_train_data, conllu_dev_data, conllu_test_data, conllu_train_data, tok_vocabulary, conllu_vocabulary, tagset, dev_t, test_t
    #return tok_train_data
    #return tok_vocabulary
    #return tagset
    #return conllu_test_data

        
	
def prep_tok_data(inp):
    final_tok_data = []
    doc_ids = []
    count = 0
    with open (inp) as f:
        tokens = []
        tags = []
        for line in f:
            if line.startswith("#"):
                if count > 0:
                    if len(tokens) == 0:
                        print ("Empty")
                    else:
                        final_tok_data.append((tokens,tags))
                #print(line.strip().split()[4])
                doc_ids.append(line.strip().split()[4])
                tokens = []
                tags = []
                count = count + 1
            elif line.strip()!="":
                #print(line.strip().split()[1])
                tokens.append(line.strip().split()[1])
                tags.append(line.strip().split()[-1])
    final_tok_data.append((tokens,tags))
    return final_tok_data

def prep_conll_data(inp):
    final_conllu_data = []
    sentence_count = 0
    count1 = 0
    with open (inp) as f:
        tokens = []
        tags = []
        for line in f:
            if line.strip() == "":
                if len(tokens) != 0:
                    final_conllu_data.append((tokens,tags))
                    tokens = []
                    tags = []
                sentence_count = sentence_count + 1
            elif line.startswith("#"):
                count1 = count1 + 1
            elif line.strip()!="":
                tokens.append(line.strip().split()[1])
                if "BeginSeg=Yes" in line.strip().split()[-1]:
                    tags.append("BeginSeg=Yes")
                elif "Seg=I-Conn" in line.strip().split()[-1]:
                    tags.append("Seg=I-Conn")
                elif "Seg=B-Conn" in line.strip().split()[-1]:
                    tags.append("Seg=B-Conn")
                else:
                    tags.append('_')
    if len(tokens) != 0:
        final_conllu_data.append((tokens,tags))
    return final_conllu_data

def vocab(data):
    words = {}
    i = 0
    for tokens, tags in data:
        for word in tokens:
            if word in words:
                i = i
            else:
                words[word] = i
                i = i + 1
    return words

def prep_seq(seq, words):
    tens = [words[w] if w in words else len(words)-1 for w in seq]
    return torch.LongTensor(tens)

def prep_label(label):
    var = torch.LongTensor([label])
    return var

tok_dev_set, tok_test_set, tok_train_set, conllu_dev_set, conllu_test_set, conllu_train_set, tok_vocab, conllu_vocab, tag_vocab, dev_file, test_file = inp_data(sys.argv[1])

def get_metrics(gold, pred):
    assert len(gold) == len(pred)
    correct = 0
    for i in range(len(gold)):
        if gold[i] == pred[i]:
            correct = correct + 1
    acc = correct/len(gold)
    precision, recall, f1, s = metrics.precision_recall_fscore_support(gold, pred)
    return acc, precision, recall, f1, s

ft_dict = { 'rand':'rand', 'eng':'./embeddings/cc.en.300.vec', 'deu':'./embeddings/cc.de.300.vec', 'eus':'./embeddings/cc.eu.300.vec', 'fas':'./embeddings/cc.fa.300.vec', 'fra':'./embeddings/cc.fr.300.vec', 'nld':'./embeddings/cc.nl.300.vec', 'por':'./embeddings/cc.pt.300.vec', 'rus':'./embeddings/cc.ru.300.vec', 'spa':'./embeddings/cc.es.300.vec', 'tur':'./embeddings/cc.tr.300.vec', 'zho':'./embeddings/cc.zh.300.vec'}

print("\nDataset name - ", sys.argv[1])

EMB_DIM = 300
HID_DIM = 100
TOK_SIZE = len(tok_vocab)
LABEL_SIZE = len(tag_vocab)
BATCH_SIZE = 1
PRE_EMB = ft_dict['rand']
VOCAB = tok_vocab

EPOCHS = 5

model = biLSTM(emb_dim=EMB_DIM, hid_dim=HID_DIM, tok_size=TOK_SIZE, label_size=LABEL_SIZE, batch_size=BATCH_SIZE, pretrained_emb=PRE_EMB, vocab=VOCAB, dropout=0.5)

loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#print(model)

#print((tok_train_set[127]))

#print(tag_vocab)

PATH = os.path.join(sys.argv[1] + '_trained.pth')

print("\nTraining...")

def train_model():
    model.train()
    for epoch in range(EPOCHS):
        print("\nEPOCH:", epoch+1)
        loss_total = 0.0
        count = 0
        for i, (tokens, tags) in tqdm(enumerate(tok_train_set)):
            count = count + 1
            model.hidden = model.init_hidden()
            optimizer.zero_grad()
            #print(tokens)
            #print(tags)
            inp = prep_seq(tokens, tok_vocab)
            #print(len(inp))
            tag = prep_seq(tags, tag_vocab)
            #print(len(tag))
            prediction = model(inp)
            #print(prediction)
            #print(prediction.shape)
            #print(tag.shape)
            loss = loss_function(prediction, tag)
            loss.backward()
            optimizer.step()
            loss_total = loss_total + loss.item()
            avg_loss = loss_total/count
        print("\nLoss = ", avg_loss)

    print('Finished Training')
    torch.save(model.state_dict(), PATH)

train_model()

print("\nTesting...")

model.load_state_dict(torch.load(PATH))

gold = []
pred = []
with torch.no_grad():
    for (tokens, tags) in (tok_dev_set):
        inp = prep_seq(tokens, tok_vocab)
        tag = prep_seq(tags, tag_vocab)
        final_pred = model(inp)
        y_pred = torch.argmax(final_pred, dim=1)
        gold.append(tag)
        pred.append(y_pred)
dev_pred = np.concatenate(pred)
dev_gold = np.concatenate(gold)
#print(len(pred))
#print(len(gold))
#print(len(dev_pred))
#print(len(dev_gold))
acc, precision, recall, f1, s = get_metrics(dev_gold, dev_pred)
print("\nDev Accuracy = ", acc)
print("\nDev Precision = ", precision[1])
print("\nDev Recall = ", recall[1])
print("\nDev F1 score = ", f1[1])

gold = []
pred = []
with torch.no_grad():
    for (tokens, tags) in (tok_test_set):
        inp = prep_seq(tokens, tok_vocab)
        tag = prep_seq(tags, tag_vocab)
        final_pred = model(inp)
        y_pred = torch.argmax(final_pred, dim=1)
        gold.append(tag)
        pred.append(y_pred)
test_pred = np.concatenate(pred)
test_gold = np.concatenate([[tok for tok in preds] for preds in gold]).reshape(test_pred.shape)
#print(gold)
#print(pred)
acc, precision, recall, f1, s = get_metrics(test_gold, test_pred)
print("\nTest Accuracy = ", acc)
print("\nTest Precision = ", precision[1])
print("\nTest Recall = ", recall[1])
print("\nTest F1 score = ", f1[1])

#print(model)
#print(len(conllu_vocab))
#print(tok_dev_set)
#print(inp_data("eng.rst.rstdt"))
#print(conllu_vocab)
#print(train_tok_model)

if 'pdtb' in sys.argv[1]:
    def print_test_output():
        output_dir = "./outputs"
        final_test_file = os.path.join(output_dir, sys.argv[1] + '_test.preds')
        l1 = open(test_file).readlines()
        with open (final_test_file, 'w') as f:
            count = 0
            for line in l1:
                line = line.strip()
                if line.startswith("#"):
                    f.write( line.strip()+'\n')
                elif line.strip()!="":
                    tag = test_pred[count]
                    if tag == 0:
                        f_tag = '_'
                    elif tag == 1: 
                        f_tag = 'Seg=B-Conn'
                    elif tag == 2:
                        f_tag = 'Seg=I-Conn'
                    f.write( '\t'.join( line.strip().split('\t')[:-1] )+'\t'+f_tag+'\n' )
                    count = count + 1
                else:
                    f.write( line.strip()+'\n')
    print_test_output()
    
    def print_dev_output():
        output_dir = "./outputs"
        final_dev_file = os.path.join(output_dir, sys.argv[1] + '_dev.preds')
        l1 = open(dev_file).readlines()
        with open (final_dev_file, 'w') as f:
            count = 0
            for line in l1:
                line = line.strip()
                if line.startswith("#"):
                    f.write( line.strip()+'\n')
                elif line.strip()!="":
                    tag = dev_pred[count]
                    if tag == 0:
                        f_tag = '_'
                    elif tag == 1: 
                        f_tag = 'Seg=B-Conn'
                    elif tag == 2:
                        f_tag = 'Seg=I-Conn'
                    f.write( '\t'.join( line.strip().split('\t')[:-1] )+'\t'+f_tag+'\n' )
                    count = count + 1
                else:
                    f.write( line.strip()+'\n')
    print_dev_output()

else:
    def print_test_output():
        output_dir = "./outputs"
        final_test_file = os.path.join(output_dir, sys.argv[1] + '_test.preds')
        l1 = open(test_file).readlines()
        with open (final_test_file, 'w') as f:
            count = 0
            for line in l1:
                line = line.strip()
                if line.startswith("#"):
                    f.write( line.strip()+'\n')
                elif line.strip()!="":
                    tag = test_pred[count]
                    if tag == 1:
                        f_tag = 'BeginSeg=Yes'
                    else: 
                        f_tag = '_'
                    f.write( '\t'.join( line.strip().split('\t')[:-1] )+'\t'+f_tag+'\n' )
                    count = count + 1
                else:
                    f.write( line.strip()+'\n')
    print_test_output()

    def print_dev_output():
        output_dir = "./outputs"
        final_dev_file = os.path.join(output_dir, sys.argv[1] + '_dev.preds')
        l2 = open(dev_file).readlines()
        with open (final_dev_file, 'w') as f:
            count = 0
            for line in l2:
                line = line.strip()
                if line.startswith("#"):
                    f.write( line.strip()+'\n')
                elif line.strip()!="":
                    tag = dev_pred[count]
                    if tag == 1:
                        f_tag = 'BeginSeg=Yes'
                    else: 
                        f_tag = '_'
                    f.write( '\t'.join( line.strip().split('\t')[:-1] )+'\t'+f_tag+'\n' )
                    count = count + 1
                else:
                    f.write( line.strip()+'\n')
    print_dev_output()
    
print("Prediction files stored in ./outputs folder")