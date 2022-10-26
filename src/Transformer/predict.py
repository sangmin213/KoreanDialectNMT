import pickle
import argparse

import torch
from soynlp.tokenizer import LTokenizer

from utils import Params, clean_text, display_attention
from model.transformer import Transformer
import nltk.translate.bleu_score as bleu
import pandas as pd


def bleu_score(sentence_list, n_gram=4):
    weights = [1./ n_gram for _ in range(n_gram)]
    
    try:
        smt_func = bleu.SmoothingFunction()
        score = 0.0
        
        for _, dia, inf in sentence_list:
            score += bleu.sentence_bleu([dia.split()],
                                        inf.split(),
                                        weights,
                                        smoothing_function=smt_func.method2)
        if len(sentence_list) == 0: 
            return 0
        else :
            return score / len(sentence_list)
    except Exception as ex:
        print(ex)
        return 0

def predict(config):
    params = Params('config/params_jj.json')
    if config.dialect == "gw":
        DIALECT = "gw"
        DIALECT2 = "jj"
    elif config.dialect == "jj":
        DIALECT2 = "gw"
        DIALECT = "jj"

    # load tokenizer and torchtext Fields
    pickle_tokenizer = open('pickles/jj/dia_tokenizer.pickle', 'rb') # input을 토큰화 하기 위함
    cohesion_scores = pickle.load(pickle_tokenizer)
    tokenizer = LTokenizer(scores=cohesion_scores) # 표준어 입력 문장 토크나이저

    pickle_std = open('pickles/jj/dia.pickle', 'rb') 
    std = pickle.load(pickle_std)
    pickle_dia = open('pickles/jj/std.pickle', 'rb')
    dia = pickle.load(pickle_dia)

    # select model and load trained model
    model = Transformer(params)
    model.load_state_dict(torch.load(params.save_model))
    model.to(params.device)
    model.eval()
    
    # input = clean_text(config.input) # --input 나는 오늘 노트북 존에 왔다.
    seq1 = []
    seq2 = []
    df = pd.read_csv('./data/gw_jj.csv')
    for idx,input in enumerate(df[f'{DIALECT}']):
        if idx == 100:
            break
        # convert input into tensor and forward it through selected model
        tokenized = tokenizer.tokenize(input) # 입력 문장 토큰화
        indexed = [std.vocab.stoi[token] for token in tokenized]

        source = torch.LongTensor(indexed).unsqueeze(0).to(params.device)  # [1, source_len]: unsqueeze to add batch size
        # target = torch.zeros(1, params.max_len).type_as(source.data)       # [1, max_len]
        target = torch.zeros(1, params.max_len).type_as(source.data)       # [1, max_len]

        encoder_output = model.encoder(source)
        next_symbol = dia.vocab.stoi['<sos>']

        pad_idx = dia.vocab.stoi['<pad>']
        unk_idx = dia.vocab.stoi['<unk>']
        eos_idx = dia.vocab.stoi['<eos>']
        # print(pad_idx, unk_idx, eos_idx, next_symbol)

        for i in range(0, params.max_len):
            target[0][i] = next_symbol
            decoder_output, _ = model.decoder(target, source, encoder_output)  # [1, target length, output dim]
            prob = decoder_output.squeeze(0).max(dim=-1, keepdim=False)[1]
            next_word = prob.data[i]
            next_symbol = next_word.item()

        try:
            eos_idx = int(torch.where(target[0] == eos_idx)[0][0])
            target = target[0][:eos_idx].unsqueeze(0)
        except:
            error = 1

        # print(target.shape)
        # print(source.shape)
        # print(eos_idx)
        # exit(1)

        # translation_tensor = [target length] filed with word indices
        target, attention_map = model(source, target)
        target = target.squeeze(0).max(dim=-1)[1]

        translated_token = [dia.vocab.itos[token] for token in target]
        # translation = translated_token[:translated_token.index('<eos>')]
        translation = translated_token
        translation = ' '.join(translation)

        seq1.append(input)
        seq2.append(translation)
        # seq.append((input,df[f'{DIALECT2}'][idx],translation))
    
    df = pd.DataFrame({f'{DIALECT}':seq1, 'std':seq2 })
    df.to_csv(f"./{DIALECT}2st_test.csv")

    # print("Now calculating blue score ...\n")
    # for i in range(1, 5):
    #     print(f'{i} {bleu_score(seq, i):.3f}')
    
    
    # print(f'std> {input}')
    # print(f'dia> {translation.capitalize()}')
        # error = 0
        # display_attention(tokenized, translated_token, attention_map[4].squeeze(0)[:-1])

    # seq.append((input,df['dialect_form'][idx],translation))
    # score = bleu_score(seq)
    # print(score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Standard-Dialect Translation prediction')
    parser.add_argument('--dialect', action='store', dest='dialect', default='gs',
                    help='Target Dialect')
    option = parser.parse_args()

    predict(option)
