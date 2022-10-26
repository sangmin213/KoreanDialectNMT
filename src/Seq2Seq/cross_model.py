import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import time
import os
import argparse
import pandas as pd

from base.data_loader import *
from base.seq2seq_attn import *
from base.helper import *
from base.inference import *
from base.train import *



# setting
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# option
parser = argparse.ArgumentParser()
parser.add_argument('--dialect', action='store', dest='dialect', default='gs',
                    help='Target Dialect')
parser.add_argument('--maxlen', action='store', dest='maxlen', type=int,
                    help='Max length of target data')
parser.add_argument('--input_level', action='store', dest='input_level', default='syl',
                    help='Input level of train data')
parser.add_argument('--model', action='store', dest='model_type', default='seq2seq',
                    help='Model_Type')
parser.add_argument('--train', action='store_true', dest='train', default=False,
                    help='Indicates if model has to be trained')
parser.add_argument('--copy', action='store_true', dest='copy', default=False,
                    help='final model copy mechanism')               
parser.add_argument('--copy1', action='store_true', dest='copy1', default=False,
                    help='copy mechanism')                    
parser.add_argument('--copy2', action='store_true', dest='copy2', default=False,
                    help='copy mechanism')                     
parser.add_argument('--jj2gw', action='store_true', dest='jj2gw', default=False,
                    help='gw2jj or jj2gw')
opt = parser.parse_args()

DIALECT = opt.dialect
INPUT_LEVEL = opt.input_level # syl, word, jaso
train_flag = opt.train
MODEL_TYPE = opt.model_type
COPY = opt.copy
COPY1 = opt.copy1
COPY2 = opt.copy2
JJ2GW = opt.jj2gw

if opt.maxlen == None:
    if INPUT_LEVEL == 'syl':
        MAX_LENGTH = 110
    elif INPUT_LEVEL == 'word':
        MAX_LENGTH = 30
    elif INPUT_LEVEL == 'jaso':
        raise NotImplementedError
else:
    MAX_LENGTH = opt.maxlen

if COPY:
    COPY = "copy_"
else:
    COPY = ""

if COPY1:
    COPY1 = "copy_"
else:
    COPY1 = ""
if COPY2:
    COPY2 = "copy_"
else:
    COPY2 = ""


path_gw_train = './data/sent_gw_train.json'
path_gw_test = './data/sent_gw_test.json'
path_jj_train = './data/sent_jj_train.json'
path_jj_test = './data/sent_jj_test.json'
path_gwjj_train = './data/sent_gwjj_train.json'
path_gwjj_test = './data/sent_gwjj_test.json'


if DIALECT == 'gw':
    PATH_TRAIN = path_gw_train
    PATH_TEST = path_gw_test
    PATH_TRAIN2 = path_jj_train
    PATH_TEST2 = path_jj_test
    DIALECT2 = 'jj'
elif DIALECT == 'jj':
    PATH_TRAIN = path_jj_train
    PATH_TEST = path_jj_test 
    PATH_TRAIN2 = path_gw_train
    PATH_TEST2 = path_gw_test   
    DIALECT2 = 'gw'
else:
    print('Invalid Dialect Error : {DIALECT}')
    exit()


# data load
train_loader = Loader(MAX_LENGTH, INPUT_LEVEL)
test_loader = Loader(MAX_LENGTH, INPUT_LEVEL)
train_loader.readJson(PATH_TRAIN)
test_loader.readJson(PATH_TEST)

train_loader2 = Loader(MAX_LENGTH, INPUT_LEVEL)
test_loader2 = Loader(MAX_LENGTH, INPUT_LEVEL)
train_loader2.readJson(PATH_TRAIN2,dia2std=False)
test_loader2.readJson(PATH_TEST2,dia2std=False)

train_loader3 = Loader(MAX_LENGTH, INPUT_LEVEL)
test_loader3 = Loader(MAX_LENGTH, INPUT_LEVEL)
train_loader3.readJson(path_gwjj_train,dia2std=JJ2GW)
test_loader3.readJson(path_gwjj_test,dia2std=JJ2GW)

if COPY1: # vocab 공유
    SRC = Vocab(train_loader.srcs + train_loader.trgs, INPUT_LEVEL, device) 
    TRG = Vocab(train_loader.trgs + train_loader.srcs, INPUT_LEVEL, device)
else:
    SRC = Vocab(train_loader.srcs, INPUT_LEVEL, device)
    TRG = Vocab(train_loader.trgs, INPUT_LEVEL, device)
SRC.build_vocab()
TRG.build_vocab()

if COPY2: # vocab 공유
    SRC2 = Vocab(train_loader2.srcs + train_loader2.trgs, INPUT_LEVEL, device) 
    TRG2 = Vocab(train_loader2.trgs + train_loader2.srcs, INPUT_LEVEL, device)
else:
    SRC2 = Vocab(train_loader2.srcs, INPUT_LEVEL, device)
    TRG2 = Vocab(train_loader2.trgs, INPUT_LEVEL, device)
SRC2.build_vocab()
TRG2.build_vocab()

if COPY: # vocab 공유
    SRC3 = Vocab(train_loader3.srcs + train_loader3.trgs, INPUT_LEVEL, device) 
    TRG3 = Vocab(train_loader3.trgs + train_loader3.srcs, INPUT_LEVEL, device)
else:
    SRC3 = Vocab(train_loader3.srcs, INPUT_LEVEL, device)
    TRG3 = Vocab(train_loader3.trgs, INPUT_LEVEL, device)
SRC3.build_vocab()
TRG3.build_vocab()

# GWJJ = pd.read_csv("./data/gw_jj.csv")

train_iterator = train_loader3.makeIterator(SRC3, TRG3, sos=True, eos=True)
test_iterator = test_loader3.makeIterator(SRC3, TRG3, sos=True, eos=True)

portion = int(len(test_iterator) * 0.5)
valid_iterator = test_iterator[:portion]
test_iterator = test_iterator[portion:]

INPUT_DIM = SRC.vocab_size
OUTPUT_DIM = TRG.vocab_size
INPUT_DIM2 = SRC2.vocab_size
OUTPUT_DIM2 = TRG2.vocab_size
ENC_EMB_DIM = 128 #256
DEC_EMB_DIM = 128 #256
ENC_HID_DIM = 128 #512
DEC_HID_DIM = 128 #512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
N_EPOCHS = 100
CLIP = 1

SRC_PAD_IDX = SRC.stoi['<pad>']
TRG_PAD_IDX = TRG.stoi['<pad>']
SOS_IDX = TRG.stoi['<sos>']
EOS_IDX = TRG.stoi['<eos>']

SRC_PAD_IDX2 = SRC2.stoi['<pad>']
TRG_PAD_IDX2 = TRG2.stoi['<pad>']
SOS_IDX2 = TRG2.stoi['<sos>']
EOS_IDX2 = TRG2.stoi['<eos>']

# Model
if COPY1:
    from seq2seq_attn_copy import *
else:
    from seq2seq_attn import *
attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, SOS_IDX, device, MAX_LENGTH).to(device)                

model_name = f's2sAttn_{COPY1}GRU_{INPUT_LEVEL}_{DIALECT}2st_{MAX_LENGTH}_{ENC_EMB_DIM}_{ENC_HID_DIM}'
model_pt_path = f'./models/{model_name}/{model_name}.pt'

model.load_state_dict(torch.load(model_pt_path))

if COPY2:
    from seq2seq_attn_copy import *
else:
    from seq2seq_attn import *
attn2 = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc2 = Encoder(INPUT_DIM2, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec2 = Decoder(OUTPUT_DIM2, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn2)

model2 = Seq2Seq(enc2, dec2, SOS_IDX2, device, MAX_LENGTH).to(device)                

model_name2 = f's2sAttn_{COPY2}GRU_{INPUT_LEVEL}_{DIALECT2}_{MAX_LENGTH}_{ENC_EMB_DIM}_{ENC_HID_DIM}'
model_pt_path2 = f'./models/{model_name2}/{model_name2}.pt'

model2.load_state_dict(torch.load(model_pt_path2))

PATH_LOG = f'./log/sent_encdec_{COPY1}{DIALECT}2{COPY2}{DIALECT2}_{MAX_LENGTH}_{ENC_EMB_DIM}_{ENC_HID_DIM}.json'

print(f"model1({DIALECT}2st) and model2(st2{DIALECT2}) is uploaded!\n")

# final model
model2.encoder = model.encoder

model_name2 = f's2sAttn_encdec_GRU_{INPUT_LEVEL}_{COPY1}{DIALECT}2{COPY2}{DIALECT2}_{MAX_LENGTH}_{ENC_EMB_DIM}_{ENC_HID_DIM}'
model_pt_path2 = f'./models/{model_name2}/{model_name2}.pt'


criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX) # Seq2Seq
optimizer = torch.optim.Adam(model2.parameters(), lr=0.0001,betas=(0.9, 0.98), eps=1e-9)

print(f'Model : {MODEL_TYPE.upper()}')
print(f'Using cuda : {torch.cuda.get_device_name(0)}')
print(f'Dialect : {DIALECT}')
print(f'Max Length : {MAX_LENGTH}')
print(f'# of train data : {len(train_iterator)}')
print(f'# of test data : {len(test_iterator)}')
print(f'# of valid data : {len(valid_iterator)}')
print(f'SRC Vocab size : {SRC3.vocab_size}')
print(f'TRG Vocab size : {TRG3.vocab_size}')
print('-' * 20)
print(f'Encoder embedding Dimension : {ENC_EMB_DIM}')
print(f'Decoder embedding Dimension : {DEC_EMB_DIM}')
print(f'Encoder Hidden Dimension : {ENC_HID_DIM}')
print(f'Decoder Hidden Dimension : {DEC_HID_DIM}')
print(f'Encoder dropout rate : {ENC_DROPOUT}')
print(f'Decoder dropout rate : {DEC_DROPOUT}')
print(f'# of epochs : {N_EPOCHS}')
print('-' * 20)
print(f'The model has {count_parameters(model2):,} trainable parameters')

try:
    if not os.path.exists(f'./models/{model_name2}'):
        os.makedirs(f'./models/{model_name2}')
except OSError:
    print(f'Failed to create directory : ./models/{model_name2}')

if train_flag == True:
    train_model(model = model2,
                model_type = MODEL_TYPE, 
                train_iterator = train_iterator, 
                valid_iterator = valid_iterator, 
                optimizer = optimizer, 
                criterion = criterion, 
                CLIP = CLIP, 
                N_EPOCHS = N_EPOCHS, 
                model_pt_path = model_pt_path2)

test_loss = evaluate(model2, MODEL_TYPE, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')    

test_pair = [[src, trg] for src, trg in zip(test_loader3.srcs, test_loader3.trgs)]
if MODEL_TYPE == 'seq2seq':
    result_dict = save_log(PATH_LOG, model2, MODEL_TYPE, SRC3, TRG3, test_pair, INPUT_LEVEL, device)
# elif MODEL_TYPE == 'transformer':
#     result_dict = save_log(PATH_LOG, model, model2, MODEL_TYPE, SRC, TRG, test_pair, INPUT_LEVEL, SOS_IDX, EOS_IDX, MAX_LENGTH, device)
result_tuple = [[d['standard'], d['dialect'], d['inference']]for d in result_dict]

print("Now calculating blue score ...\n")
for i in range(1, 5):
    print(f'{i} {bleu_score(result_tuple, i):.3f}')



# print("Interactive Mode start!")
# while True:
#     src = input('>>> ')
#     if src == '':
#         break 
#     if MODEL_TYPE == 'seq2seq':
#         inf = translate_sentence(model2, MODEL_TYPE, SRC3, TRG3, src, INPUT_LEVEL, device) # 소스방언 -> 표준어
#         # inf2 = translate_sentence(model2, MODEL_TYPE, SRC2, TRG2, inf, INPUT_LEVEL, device) # 표준어 -> 타겟방언
#     # if MODEL_TYPE == 'transformer':
#     #     inf = translate_sentence(model, MODEL_TYPE, SRC, TRG, src, INPUT_LEVEL, SOS_IDX, EOS_IDX, MAX_LENGTH, device)
#     print(f'<<< {inf}')
#     # print(f'<<< {inf2}')