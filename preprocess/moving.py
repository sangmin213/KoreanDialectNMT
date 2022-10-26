import os
import pandas as pd
import json

location=['강원도','경상도','전라도','제주도','충청도']

GangWonlist=[]
GyeongSanglist=[]
JeonLalist=[]
JeJulist=[]
ChungCheonglist=[]
listname=[GangWonlist,GyeongSanglist,JeonLalist,JeJulist,ChungCheonglist]


for loc,location_list in zip(location,listname):
    path=f'./{loc}/'
    cnt=1
    for file_name in os.listdir(path):
        location_list.append(path+file_name)
        if cnt == 600: # 각 방언 데이터 1000개씩 확인
            break
        cnt+=1
        

for index in [0,3]: # 강원도 / 제주도 데이터
    first=1
    for idx,path in enumerate(listname[index]):
        # json 파일 읽어들이기
        with open(path,encoding='utf-8-sig') as json_data:
            data = json.load(json_data)
        # 첫 번째 들어온 파일에 대해서는 dataframe 생성
        if first==1:
            df_utterance=pd.DataFrame(data['utterance'])
            # df_speaker=pd.DataFrame(data['speaker'])
            first=0
        # 그 이후 부터는 밑에 붙여나감
        else:
            tmp_utter=pd.DataFrame(data['utterance'])
            # tmp_speak=pd.DataFrame(data['speaker'])
            df_utterance=pd.concat([df_utterance,tmp_utter],ignore_index=True)
            # df_speaker=pd.concat([df_speaker,tmp_speak],ignore_index=True)

    df = df_utterance.loc[:,['standard_form','dialect_form']].copy()
    if index==0:
        df.to_csv('./GW.csv')
    else:
        df.to_csv('./JJ.csv')
    