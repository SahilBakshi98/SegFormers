import enum
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time, random
import os, sys
from tqdm import tqdm
import numpy as np
import argparse
from sklearn import metrics
from transformers import BertTokenizerFast
from transformers import BertForTokenClassification, AdamW
from torch.utils.data import DataLoader
from transformers import logging

#torch.set_num_threads(6)
torch.manual_seed(1)
logging.set_verbosity_error()

data_folder = "../../data"
split_folder = "./split_docs"
   
def inp_data(data_name):
    dev_t = os.path.join(data_folder, data_name, data_name + '_dev.tok')
    test_t = os.path.join(data_folder, data_name, data_name + '_test.tok')
    train_t = os.path.join(data_folder, data_name, data_name + '_train.tok')
    dev_c = os.path.join(data_folder, data_name, data_name + '_dev.conllu')
    test_c = os.path.join(data_folder, data_name, data_name + '_test.conllu')
    train_c = os.path.join(data_folder, data_name, data_name + '_train.conllu')
    
    dev_st = os.path.join(split_folder, data_name, data_name + '_dev_split.tok')
    test_st = os.path.join(split_folder, data_name, data_name + '_test_split.tok')
    train_st = os.path.join(split_folder, data_name, data_name + '_train_split.tok')
    
    if os.path.isfile(dev_t) == False or os.path.isfile(test_t) == False or os.path.isfile(train_t) == False or os.path.isfile(dev_c) == False or os.path.isfile(test_c) == False or os.path.isfile(train_c) == False:
        print ("File does not exist", file=sys.stderr)

    tok_dev_data = prep_tok_data(dev_st)
    tok_test_data = prep_tok_data(test_st)
    tok_train_data = prep_tok_data(train_st)
    conllu_dev_data = prep_conll_data(dev_c)
    conllu_test_data = prep_conll_data(test_c)
    conllu_train_data = prep_conll_data(train_c)
    
    tok_vocabulary = vocab(tok_train_data)
    conllu_vocabulary = vocab(conllu_train_data)

    if "pdtb" in data_name:
        tagset = {'Seg=B-Conn':0, 'Seg=I-Conn':1, '_':2}
    else:
        tagset = {'BeginSeg=Yes':0, '_':1}
    
    return tok_dev_data, tok_test_data, tok_train_data, conllu_dev_data, conllu_test_data, conllu_train_data, tok_vocabulary, conllu_vocabulary, tagset, dev_t, test_t
    #return tok_train_data
    #return tok_vocabulary
    #return tagset
    #return conllu_test_data

def prep_tok_data(inp):
    final_tok_data = []
    doc_ids = []
    count = 0
    sentence_count = 0
    with open (inp) as f:
        tokens = []
        tags = []
        for line in f:
            if line.strip() == "":
                if len(tokens) != 0:
                    final_tok_data.append((tokens,tags))
                    tokens = []
                    tags = []
                sentence_count = sentence_count + 1
            elif line.startswith("#"):
                count = count + 1
            elif line.strip()!="":
                #print(line.strip().split()[1])
                tokens.append(line.strip().split()[1])
                if "BeginSeg=Yes" in line.strip().split()[-1]:
                    tags.append("BeginSeg=Yes")
                elif "Seg=I-Conn" in line.strip().split()[-1]:
                    tags.append("Seg=I-Conn")
                elif "Seg=B-Conn" in line.strip().split()[-1]:
                    tags.append("Seg=B-Conn")
                else:
                    tags.append('_')
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

#print(conllu_dev_set[0][0][10:17], conllu_dev_set[0][1][10:17])
#print(tag_vocab)
#print(tok_dev_set[0])

print("\nDataset name - ", sys.argv[1])
print("\nFile type - .tok")

train_texts = []
dev_texts = []
test_texts = []
train_tags = []
dev_tags = []
test_tags = []
mask_dev = []
mask_test = []

for i, (tokens,tags) in enumerate(tok_train_set):
    train_texts.append(tokens)
    train_tags.append(tags)

for i, (tokens,tags) in enumerate(tok_dev_set):
    dev_texts.append(tokens)
    dev_tags.append(tags)

for i, (tokens,tags) in enumerate(tok_test_set):
    test_texts.append(tokens)
    test_tags.append(tags)
    
for tag in (dev_tags):
    mask_dev.append(len(tag))
    
for tag in (test_tags):
    mask_test.append(len(tag))

#print(len(train_texts), len(dev_texts), len(test_texts))
#print(train_texts[0][0:5])
#print(train_tags[0])
#print((train_texts[0][-1]))
#print(train_tags)

#unique_tags = set(tag for doc in train_tags for tag in doc)
tag2id = tag_vocab
id2tag = {id: tag for tag, id in tag2id.items()}

#print(unique_tags)
#print(len(tag2id))
#print(id2tag)
print("\nPreparing and tokenizing data...")

tok_mask_length = 200

tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding =True, truncation=True)
dev_encodings = tokenizer(dev_texts, is_split_into_words=True, return_offsets_mapping=True, padding =True, truncation=True)
test_encodings = tokenizer(test_texts, is_split_into_words=True, return_offsets_mapping=True, padding =True, truncation=True)

#print(len(tag_vocab))
#print((train_encodings[0]))
#print(tokenizer("Hello World")['input_ids'])
flag = 0

def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        #print(len(doc_labels))
        #print(len(doc_offset))
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)
        #print(arr_offset)

        # set labels whose first offset position is 0 and the second is not 0
        #print(doc_enc_labels)
        if (len(doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)]) == len(doc_labels)):
            doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
            #print(len(doc_enc_labels))
        else:
            flag = 1
            len_diff = len(doc_labels) - len(doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)])
            #print(len_diff)
            #temp = doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)]
            #print(len(arr_offset))
            for i in range(len_diff):
                for j in range(len(arr_offset)):
                    '''if arr_offset[(len(arr_offset))-j-1][1] != 0 and j != 0 and arr_offset[(len(arr_offset))-j][1] == 0:
                        #print(arr_offset)
                        arr_offset[(len(arr_offset))-j][1] = 1
                        print("FIRSTTTTTT IFFFFFFFFFF")
                        break
                    elif arr_offset[(len(arr_offset))-j-1][1] != 0 and j == 0 and arr_offset[0][1] == 0:
                        arr_offset[0][1] = 1
                        print("SECONDDDDDDDDD")
                        break
                        #print(len(arr_offset))
                    elif arr_offset[j][0] != 0 and arr_offset[j][1] != 0:
                        arr_offset[j][0] = 0
                        print("THIRDDDDDHHHHHHHHHH")
                        break'''
                    if arr_offset[j][0] != 0 and arr_offset[j][1] != 0:
                        arr_offset[j][0] = 0
                        #print("FIRST")
                        break
                    elif arr_offset[j][0] == 0 and arr_offset[j][1] == 0:
                        arr_offset[j][1] = 1
                        #print("SECOND")
                        break
            #print(len(doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)]))
            #print(len(doc_labels))
            #temp = doc_labels
            #doc_enc_labels = temp
            #print(len(arr_offset))
            #print(arr_offset[1][0])
            doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
            #print(len(doc_enc_labels))
        #print(doc_enc_labels)
        #print(len(doc_labels))
        #print(len(doc_enc_labels))
        encoded_labels.append(doc_enc_labels)
        #print(len(encoded_labels[0]))

    return encoded_labels

train_labels = encode_tags(train_tags, train_encodings)
dev_labels = encode_tags(dev_tags, dev_encodings)
test_labels = encode_tags(test_tags, test_encodings)

#print(len(dev_labels[10]))

'''train_labels = []
for i in range(len(train_tags)):
    temp = [0]*(tok_mask_length - len(train_tags[i]))
    labels = [tag2id[tag] for tag in train_tags[i]]
    labels.extend(temp)
    train_labels.append(labels)

#print(train_labels[1])

dev_labels = []
for i in range(len(dev_tags)):
    temp = [0]*(tok_mask_length - len(dev_tags[i]))
    labels = [tag2id[tag] for tag in dev_tags[i]]
    labels.extend(temp)
    dev_labels.append(labels)

test_labels = []
for i in range(len(test_tags)):
    temp = [0]*(tok_mask_length - len(test_tags[i]))
    labels = [tag2id[tag] for tag in test_tags[i]]
    labels.extend(temp)
    test_labels.append(labels)

#print((train_labels[-1]))'''

class CustDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        #print("ITEM", item['labels'].shape, item['input_ids'].shape)
        return item

    def __len__(self):
        return len(self.labels)
    
train_encodings.pop("offset_mapping")
dev_encodings.pop("offset_mapping")
test_encodings.pop("offset_mapping")

train_dataset = CustDataset(train_encodings, train_labels)
dev_dataset = CustDataset(dev_encodings, dev_labels)
test_dataset = CustDataset(test_encodings, test_labels)

#print(train_dataset[34])

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')

print("\nDevice - ", device)

model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased', num_labels = len(tag_vocab))

#print(model)
#print("Before model.to")
model.train().to(device)

print("\nLoading data...")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

#print("After loader")
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

print("\nTraining...")
epochs = 5

PATH = os.path.join(sys.argv[1] + '_tok_trained.pth')

def train_model():
    for epoch in tqdm(range(epochs)):
        print("\nEPOCH:", epoch+1)
        loss_total = 0.0
        count = 0
        for batch in tqdm(train_loader):
            count = count + 1
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            loss_total = loss_total + loss.item()
            avg_loss = loss_total/count
        print("\nLoss = ", avg_loss)

    print('Finished Training')
    torch.save(model.state_dict(), PATH)
    
train_model()

model.load_state_dict(torch.load(PATH))

print('\nTesting...')

model.eval()

dev_gold = []
dev_pred = []
false_true = []
dev_loader = DataLoader(dev_dataset, batch_size=1)
for i, batch in enumerate(dev_loader):
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        s_lengths = batch['attention_mask'].sum(dim=1)
        #print(outputs[1])
        #print("s_length", s_lengths)
        for idx, length in enumerate(s_lengths):
            #length = mask_dev[i]
            true_values = batch['labels'][idx][:length]
            #print("true", true_values)
            pred_values = torch.argmax(outputs[1].sigmoid().detach().cpu(), dim=2)[idx][:length]
            #print("pred", pred_values)
            for i in range(len(true_values)):
                if true_values[i] == -100:
                    false_true.append(true_values[i])
                else:
                    dev_gold.append(true_values[i].item())
                    dev_pred.append(pred_values[i].item())
        #gold.append(true_values)
        #pred.append(pred_values)
#dev_pred = np.concatenate(pred)
#dev_gold = np.concatenate(gold)
#print((dev_gold[0]))
#print((dev_pred[0]))
#print(len(dev_gold))
#print(len(dev_pred))
#print(dev_pred)

gold_count = 0
for tag in dev_gold:
    if tag == 0:
        gold_count = gold_count + 1
        
dev_count = 0
for tag in dev_pred:
    if tag == 0:
        dev_count = dev_count + 1
        
#print(gold_count)
#print(dev_count)

acc, precision, recall, f1, s = get_metrics(dev_gold, dev_pred)
print("\nDev Accuracy = ", acc)
#print("\nDev Precision = ", precision[1])
#print("\nDev Recall = ", recall[1])
#print("\nDev F1 score = ", f1[1])

test_gold = []
test_pred = []
false_true = []
test_loader = DataLoader(test_dataset, batch_size=1)
for i, batch in enumerate(test_loader):
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        s_lengths = batch['attention_mask'].sum(dim=1)
        for idx, length in enumerate(s_lengths):
            #length = mask_test[i]
            true_values = batch['labels'][idx][:length]
            pred_values = torch.argmax(outputs[1].cpu(), dim=2)[idx][:length]
            for i in range(len(true_values)):
                if true_values[i] == -100:
                    false_true.append(true_values[i])
                else:
                    test_gold.append(true_values[i].item())
                    test_pred.append(pred_values[i].item())
#print(gold)
#print(test_pred)
acc, precision, recall, f1, s = get_metrics(test_gold, test_pred)
print("\nTest Accuracy = ", acc)
#print("\nTest Precision = ", precision[1])
#print("\nTest Recall = ", recall[1])
#print("\nTest F1 score = ", f1[1])

if 'pdtb' in sys.argv[1]:
    def print_test_output():
        output_dir = "./outputs"
        final_test_file = os.path.join(output_dir, sys.argv[1] + '_test.tok.preds')
        l1 = open(test_file).readlines()
        with open (final_test_file, 'w') as f:
            count = 0
            for line in l1:
                line = line.strip()
                if line.startswith("#"):
                    f.write( line.strip()+'\n')
                elif line.strip()!="":
                    tag = test_pred[count]
                    f_tag = id2tag[tag]
                    f.write( '\t'.join( line.strip().split('\t')[:-1] )+'\t'+f_tag+'\n' )
                    count = count + 1
                else:
                    f.write( line.strip()+'\n')
    print_test_output()
    
    def print_dev_output():
        output_dir = "./outputs"
        final_dev_file = os.path.join(output_dir, sys.argv[1] + '_dev.tok.preds')
        l1 = open(dev_file).readlines()
        with open (final_dev_file, 'w') as f:
            count = 0
            for line in l1:
                line = line.strip()
                if line.startswith("#"):
                    f.write( line.strip()+'\n')
                elif line.strip()!="":
                    tag = dev_pred[count]
                    f_tag = id2tag[tag]
                    f.write( '\t'.join( line.strip().split('\t')[:-1] )+'\t'+f_tag+'\n' )
                    count = count + 1
                else:
                    f.write( line.strip()+'\n')
    print_dev_output()

else:
    def print_test_output():
        output_dir = "./outputs"
        final_test_file = os.path.join(output_dir, sys.argv[1] + '_test.tok.preds')
        l1 = open(test_file).readlines()
        with open (final_test_file, 'w') as f:
            count = 0
            for line in l1:
                line = line.strip()
                if line.startswith("#"):
                    f.write( line.strip()+'\n')
                elif line.strip()!="":
                    tag = test_pred[count]
                    f_tag = id2tag[tag]
                    f.write( '\t'.join( line.strip().split('\t')[:-1] )+'\t'+f_tag+'\n' )
                    count = count + 1
                else:
                    f.write( line.strip()+'\n')
    print_test_output()

    def print_dev_output():
        output_dir = "./outputs"
        final_dev_file = os.path.join(output_dir, sys.argv[1] + '_dev.tok.preds')
        l2 = open(dev_file).readlines()
        with open (final_dev_file, 'w') as f:
            count = 0
            for line in l2:
                line = line.strip()
                if line.startswith("#"):
                    f.write( line.strip()+'\n')
                elif line.strip()!="":
                    tag = dev_pred[count]
                    f_tag = id2tag[tag]
                    f.write( '\t'.join( line.strip().split('\t')[:-1] )+'\t'+f_tag+'\n' )
                    count = count + 1
                else:
                    f.write( line.strip()+'\n')
    print_dev_output()
      
print("Prediction files stored in ./outputs folder")