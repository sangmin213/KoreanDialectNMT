import os
import pickle
import argparse

import pandas as pd
from pathlib import Path
from utils import convert_to_dataset

from torchtext import data as ttd
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer


def build_tokenizer():
    """
    Train soynlp tokenizer which will be used to tokenize Korean input sentence
    """
    print(f'Now building soy-nlp tokenizer . . .')

    data_dir = Path().cwd() / 'data'
    train_file = os.path.join(data_dir, 'gw-st.csv')

    df = pd.read_csv(train_file, encoding='utf-8')

    # if encounters non-text row, we should skip it
    std_lines = [row.standard_form
                 for _, row in df.iterrows() if type(row.standard_form) == str]
    
    # 최소 등장 횟수가 5인 단어를 추출하는 클래스 정의
    word_extractor = WordExtractor(min_frequency=5)
    word_extractor.train(std_lines) # 표준어 데이터를 가지고 학습시킴

    # 표준어 데이터로 만들어진 추출기에서 빈도수에 따라 각 단어 별 점수가 정의됨. ex) '나는' = 0.44433809453688705
    # 코드로는 다음과 같이 확인 가능함.
    # file_kor = open('pickles/tokenizer.pickle', 'rb')
    # kor = pickle.load(file_kor)
    # print(kor['나는']) // 코드 예시
    word_scores = word_extractor.extract() 
    cohesion_scores = {word: score.cohesion_forward
                       for word, score in word_scores.items()}

    with open('pickles/std_tokenizer.pickle', 'wb') as pickle_out:
        pickle.dump(cohesion_scores, pickle_out)

    '''위의 과정을 방언에 대해서도 똑같이 적용해서 토크나이저 생성'''
    dia_lines = [row.dialect_form
                 for _, row in df.iterrows() if type(row.dialect_form) == str]

    word_extractor = WordExtractor(min_frequency=5)
    word_extractor.train(dia_lines) # 방언 데이터를 가지고 학습시킴

    word_scores = word_extractor.extract() 
    cohesion_scores = {word: score.cohesion_forward
                       for word, score in word_scores.items()}

    with open('pickles/dia_tokenizer.pickle', 'wb') as pickle_out:
        pickle.dump(cohesion_scores, pickle_out)


def build_vocab(config):
    """
    Build vocab used to convert input sentence into word indices using soynlp and spacy tokenizer
    Args:
        config: configuration containing various options
    """
    std_pickle_tokenizer = open('pickles/std_tokenizer.pickle', 'rb')
    cohesion_scores = pickle.load(std_pickle_tokenizer)
    std_tokenizer = LTokenizer(scores=cohesion_scores)

    dia_pickle_tokenizer = open('pickles/dia_tokenizer.pickle', 'rb')
    cohesion_scores = pickle.load(dia_pickle_tokenizer)
    dia_tokenizer = LTokenizer(scores=cohesion_scores)


    # include lengths of the source sentences to use pack pad sequence
    std = ttd.Field(tokenize=std_tokenizer.tokenize,
                    lower=True,
                    init_token='<sos>',
                    eos_token='<eos>',
                    pad_token='<pad>',
                    unk_token='<unk>',
                    batch_first=True)

    dia = ttd.Field(tokenize=dia_tokenizer.tokenize,
                    lower=True,
                    init_token='<sos>',
                    eos_token='<eos>',
                    pad_token='<pad>',
                    unk_token='<unk>',
                    batch_first=True)

    # eng = ttd.Field(tokenize='spacy',
    #                 init_token='<sos>',
    #                 eos_token='<eos>',
    #                 lower=True,
    #                 batch_first=True)

    data_dir = Path().cwd() / 'data'
    train_file = os.path.join(data_dir, 'train.csv')
    train_data = pd.read_csv(train_file, encoding='utf-8')
    train_data = convert_to_dataset(train_data.loc[:,['standard_form','dialect_form']], std, dia)

    print(f'Build vocabulary using torchtext . . .')

    std.build_vocab(train_data, max_size=config.std_vocab)
    dia.build_vocab(train_data, max_size=config.dia_vocab)

    print(f'Unique tokens in Standard vocabulary: {len(std.vocab)}')
    print(f'Unique tokens in Dialect vocabulary: {len(dia.vocab)}')

    print(f'Most commonly used Standard words are as follows:')
    print(std.vocab.freqs.most_common(20))

    print(f'Most commonly used Dialect words are as follows:')
    print(dia.vocab.freqs.most_common(20))

    with open('pickles/std.pickle', 'wb') as std_file:
        pickle.dump(std, std_file)

    with open('pickles/dia.pickle', 'wb') as dia_file:
        pickle.dump(dia, dia_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pickle Builder')

    parser.add_argument('--std_vocab', type=int, default=33000)
    parser.add_argument('--dia_vocab', type=int, default=33000)

    config = parser.parse_args()

    build_tokenizer()
    build_vocab(config)
