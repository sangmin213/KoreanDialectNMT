import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import time
import os
import argparse

from data_loader import *
from seq2seq_attn import *
# from seq2seq_attn_copy import *
from inference import *
# from seq2seq_LSTM import *
# from inference_transformer import *
from helper import *
from train import *
# from train_transformer import *
# from transformer import *
from torch_transformer import *


torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                    help='Indicates if model has copy mechanism')             
parser.add_argument('--dia2std', action='store_true', dest='dia2std', default=False,
                    help='Indicates if model translate dia2std or std2dia')
opt = parser.parse_args()

# Max Word Length : 30
# Max Sly Length : 138
DIALECT = opt.dialect
INPUT_LEVEL = opt.input_level # syl, word, jaso
train_flag = opt.train
MODEL_TYPE = opt.model_type
COPY = opt.copy
DIA2STD = opt.dia2std

if COPY:
    copy_='copy_'
else:
    copy_=''
if DIA2STD:
    dia2std_='2st'
else:
    dia2std_=''

if opt.maxlen == None:
    if INPUT_LEVEL == 'syl':
        MAX_LENGTH = 110
    elif INPUT_LEVEL == 'word':
        MAX_LENGTH = 30
    elif INPUT_LEVEL == 'jaso':
        raise NotImplementedError
else:
    MAX_LENGTH = opt.maxlen


path_gs_train = './data/sent_gs_train.json'
path_gs_test = './data/sent_gs_test.json'
path_jl_train = './data/sent_jl_train.json'
path_jl_test = './data/sent_jl_test.json'
path_gw_train = './data/sent_gw_train.json'
path_gw_test = './data/sent_gw_test.json'
path_jj_train = './data/sent_jj_train.json'
path_jj_test = './data/sent_jj_test.json'
path_gwjj_train = './data/sent_gwjj_train.json'
path_gwjj_test = './data/sent_gwjj_test.json'

if DIALECT == 'gs':
    PATH_TRAIN = path_gs_train
    PATH_TEST = path_gs_test
elif DIALECT == 'jl':
    PATH_TRAIN = path_jl_train
    PATH_TEST = path_jl_test
elif DIALECT == 'gw':
    PATH_TRAIN = path_gw_train
    PATH_TEST = path_gw_test
elif DIALECT == 'jj':
    PATH_TRAIN = path_jj_train
    PATH_TEST = path_jj_test    
elif DIALECT == 'gwjj':
    PATH_TRAIN = path_gwjj_train
    PATH_TEST = path_gwjj_test    
else:
    print('Invalid Dialect Error : {DIALECT}')
    exit()

train_loader = Loader(MAX_LENGTH, INPUT_LEVEL)
test_loader = Loader(MAX_LENGTH, INPUT_LEVEL)
train_loader.readJson(PATH_TRAIN,dia2std=DIA2STD,copy=COPY)
test_loader.readJson(PATH_TEST,dia2std=DIA2STD,copy=COPY)

if COPY: # vocab 공유
    SRC = Vocab(train_loader.srcs + train_loader.trgs, INPUT_LEVEL, device) 
    TRG = Vocab(train_loader.trgs + train_loader.srcs, INPUT_LEVEL, device)
else:
    SRC = Vocab(train_loader.srcs, INPUT_LEVEL, device)
    TRG = Vocab(train_loader.trgs, INPUT_LEVEL, device)
SRC.build_vocab()
TRG.build_vocab()

# from collections import defaultdict
# dict1 = defaultdict(int)
# dict2 = defaultdict(int)
# for src, trg in zip(train_loader.srcs, train_loader.trgs):
#     src_spl = src.split()
#     trg_spl = trg.split()
#     assert len(src_spl) == len(trg_spl), 'Length is different!'
#     dict1[len(src_spl)] += 1
#     dict2[len(trg_spl)] += 1
# for i in range(410):
#     print(i, dict1[i], dict2[i])

train_iterator = train_loader.makeIterator(SRC, TRG, sos=True, eos=True)
test_iterator = test_loader.makeIterator(SRC, TRG, sos=True, eos=True)

portion = int(len(test_iterator) * 0.5)
valid_iterator = test_iterator[:portion]
test_iterator = test_iterator[portion:]

INPUT_DIM = SRC.vocab_size
OUTPUT_DIM = TRG.vocab_size
ENC_EMB_DIM = 128 #256
DEC_EMB_DIM = 128 #256
ENC_HID_DIM = 128 #512
DEC_HID_DIM = 128 #512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
N_EPOCHS = 50
CLIP = 1

SRC_PAD_IDX = SRC.stoi['<pad>']
TRG_PAD_IDX = TRG.stoi['<pad>']
SOS_IDX = TRG.stoi['<sos>']
EOS_IDX = TRG.stoi['<eos>']

''' Seq 2 Seq '''
if MODEL_TYPE =='seq2seq':
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, SOS_IDX, device, MAX_LENGTH).to(device)                
    ## model = nn.DataParallel(model)
    model.apply(init_weights)
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX) # Seq2Seq

    model_name = f's2sAttn_{copy_}GRU_{INPUT_LEVEL}_{DIALECT}{dia2std_}_{MAX_LENGTH}_{ENC_EMB_DIM}_{ENC_HID_DIM}'
    model_pt_path = f'./models/{model_name}/{model_name}.pt'

    PATH_LOG = f'./log/sent_{copy_}{INPUT_LEVEL}_{DIALECT}{dia2std_}_{MAX_LENGTH}_{ENC_EMB_DIM}_{ENC_HID_DIM}.json'
''''''

''' Transformer '''
if MODEL_TYPE == 'transformer':
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dropout = 0.10
    forward_expansion = 4
    ENC_EMB_DIM = 512 #256
    DEC_HID_DIM = 2048 #512
    
    model = Seq2SeqTransformer(num_encoder_layers, num_decoder_layers, ENC_EMB_DIM,
                                 num_heads, INPUT_DIM, OUTPUT_DIM, DEC_HID_DIM, 0.1, device)

    # model = Transformer(ENC_EMB_DIM, INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX,
    #                     num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, dropout, MAX_LENGTH, device)
    torch.manual_seed(0)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model.to(device)
    # Transformer has same embedding dimension for Encoder & Decoder
    criterion = nn.CrossEntropyLoss(ignore_index = SRC_PAD_IDX) # Transformer

    model_name = f'pytorch_Transformer_{INPUT_LEVEL}_{DIALECT}_{MAX_LENGTH}_{ENC_EMB_DIM}_{ENC_HID_DIM}_epoch_{N_EPOCHS}'
    model_pt_path = f'./models/{model_name}/{model_name}.pt'

    PATH_LOG = f'./log/sent_pytorch_Transformer_{INPUT_LEVEL}_{DIALECT}_{MAX_LENGTH}_{ENC_EMB_DIM}_{ENC_HID_DIM}_epoch_{N_EPOCHS}.json'
''''''

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,betas=(0.9, 0.98), eps=1e-9)

print(f'Model : {MODEL_TYPE.upper()}')
print(f'Using cuda : {torch.cuda.get_device_name(0)}')
print(f'Dialect : {DIALECT}')
print(f'Max Length : {MAX_LENGTH}')
print(f'# of train data : {len(train_iterator)}')
print(f'# of test data : {len(test_iterator)}')
print(f'# of valid data : {len(valid_iterator)}')
print(f'SRC Vocab size : {SRC.vocab_size}')
print(f'TRG Vocab size : {TRG.vocab_size}')
print('-' * 20)
print(f'Encoder embedding Dimension : {ENC_EMB_DIM}')
print(f'Decoder embedding Dimension : {DEC_EMB_DIM}')
print(f'Encoder Hidden Dimension : {ENC_HID_DIM}')
print(f'Decoder Hidden Dimension : {DEC_HID_DIM}')
print(f'Encoder dropout rate : {ENC_DROPOUT}')
print(f'Decoder dropout rate : {DEC_DROPOUT}')
print(f'# of epochs : {N_EPOCHS}')
print('-' * 20)
print(f'The model has {count_parameters(model):,} trainable parameters')

try:
    if not os.path.exists(f'./models/{model_name}'):
        os.makedirs(f'./models/{model_name}')
except OSError:
    print(f'Failed to create directory : ./models/{model_name}')

if train_flag == True:
    train_model(model = model,
                model_type = MODEL_TYPE, 
                train_iterator = train_iterator, 
                valid_iterator = valid_iterator, 
                optimizer = optimizer, 
                criterion = criterion, 
                CLIP = CLIP, 
                N_EPOCHS = N_EPOCHS, 
                model_pt_path = model_pt_path)


model.load_state_dict(torch.load(model_pt_path))

test_loss = evaluate(model, MODEL_TYPE, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')    

test_pair = [[src, trg] for src, trg in zip(test_loader.srcs, test_loader.trgs)]
if MODEL_TYPE == 'seq2seq':
    result_dict = save_log(PATH_LOG, model, MODEL_TYPE, SRC, TRG, test_pair[portion:], INPUT_LEVEL, device)
elif MODEL_TYPE == 'transformer':
    result_dict = save_log(PATH_LOG, model, MODEL_TYPE, SRC, TRG, test_pair[portion:], INPUT_LEVEL, SOS_IDX, EOS_IDX, MAX_LENGTH, device)
result_tuple = [[d['standard'], d['dialect'], d['inference']]for d in result_dict]

for i in range(1, 5):
    print(f'{i} {bleu_score(result_tuple, i):.3f}')

print("Interactive Mode start!")
# while True:
#     src = input('>>> ')
#     if src == '':
#         break 
#     if MODEL_TYPE == 'seq2seq':
#         inf = translate_sentence(model, MODEL_TYPE, SRC, TRG, src, INPUT_LEVEL, device)
#     if MODEL_TYPE == 'transformer':
#         inf = translate_sentence(model, MODEL_TYPE, SRC, TRG, src, INPUT_LEVEL, SOS_IDX, EOS_IDX, MAX_LENGTH, device)
#     print(f'<<< {inf}')