import pandas as pd
import re

GW = pd.read_csv('./GWnJJ/gw_jj.csv')

Data = GW[['gw','jj']].copy()

# 특수 문자 삭제하기
length = len(Data['standard_form'])
for idx,std in enumerate(Data['standard_form']):
    if idx % 100 == 0:
        print(f"1차 {idx+1}/{length} {(idx+1)*100/length:.1f}%",end="")
    dia = Data['dialect_form'].iloc[idx]
    # 특수문자 삭제
    std = re.sub('[^\w\s가-힣]','',str(std))
    dia = re.sub('[^\w\s가-힣]','',str(dia))
    # 이중 띄어쓰기 하나로 고치기
    std = re.sub('  ',' ',std)
    dia = re.sub('  ',' ',dia)
    # laughing 삭제
    std = std.replace('laughing','')
    dia = dia.replace('laughing','')
    # 영어 들어있는 문장 삭제 (name, xxx 등이 들어가서 문장의 흐름 방해)
    if re.findall('[a-zA-Z0-9]',std) or re.findall('[a-zA-Z0-9]',dia):
        std, dia = '', ''
    Data['standard_form'].iloc[idx] = std
    Data['dialect_form'].iloc[idx] = dia

# short sentence drop
short= []
maxlen=0
length = len(Data['standard_form'])
for idx,seq in enumerate(Data['standard_form']):
    if idx % 100 == 0:
        print(f"2차 {idx+1}/{length} {(idx+1)*100/length:.1f}%",end="")
    if len(seq)<6:
        Data.drop(labels=idx,inplace=True)
    if len(seq)>maxlen:
        maxlen=len(seq)

print("max_len:",maxlen)
print(len(short))
Data.reset_index(drop=True,inplace=True)
Data.to_csv("./GWnJJ/gw-st.csv")

trainlen = int(len(Data)*0.9) # 90% train data
val_len = int(len(Data)*0.95) # 5% val, test each

train = Data.iloc[:trainlen].copy()
test = Data.iloc[val_len:].copy()
test.reset_index(drop=True,inplace=True)
print(test.head())
print()

valid = Data.iloc[trainlen:val_len].copy()
valid.reset_index(drop=True,inplace=True)
print(valid.head())

train.to_csv("./GWnJJ/train.csv")
valid.to_csv("./GWnJJ/valid.csv")
test.to_csv("./GWnJJ/test.csv")

train.rename(columns = {'gw':'standard','jj':'dialect'},inplace=True)
valid.rename(columns = {'gw':'standard','jj':'dialect'},inplace=True)
test.rename(columns = {'gw':'standard','jj':'dialect'},inplace=True)

train.to_json("./korean-standard-to-dialect/data/sent_gwjj_train.json",force_ascii=False)
valid.to_json("./korean-standard-to-dialect/data/sent_gwjj_valid.json",force_ascii=False)
test.to_json("./korean-standard-to-dialect/data/sent_gwjj_test.json",force_ascii=False)

