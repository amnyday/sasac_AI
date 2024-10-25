#[1] 피드포워드(MLP) 사용해서 이름의 국적을 추론하는 뉴럴넷 만들기
#1. 데이터셋만들기, 데이터전처리
import os
import glob
import torch
from collections import defaultdict
import torch.nn as nn

#운영체제에 맞게 자동으로 경로 구분자를 처리하도록 파이썬에서는 경로를 다룰 때 os 모듈의 os.path 또는 pathlib 모듈을 사용하는 것이 좋습니다.
from pathlib import Path

#파일불러올 경로 지정
#files = glob.glob('C:/Users/user/Desktop/sasac_AI/Deep_learning/rnn중간실습(이름으로국적찾아내기)_feedforward/names/*.txt')
file_path = Path('C:/Users/user/Desktop/sasac_AI/Deep_learning/rnn중간실습(이름으로국적찾아내기)_feedforward/names')

# glob을 사용할 때는 Path 객체를 문자열로 변환하고 패턴을 추가
files = glob.glob(str(file_path / '*.txt'))  # '/' 연산자로 경로와 패턴을 결합

#파일갯수 맞는지 확인
assert len(files) == 18

#분류할 기준이 되는 내용(알파벳) 정의
all_letters = 'abcdefghijklmnopqrstuvwxyz'
#알파벳 갯수 저장해두기
num_letters = len(all_letters) # 알파벳 수 (a-z)
#알파벳 갯수 맞는지 확인
assert num_letters == 26

# 국가별 이름 데이터를 저장할 딕셔너리
category_names = {}
# 모든 국가 카테고리를 저장할 리스트
all_categories = []

#이름가져온 데이터셋, 국적가져온 데이터셋 만들기
#names 파일 열어서(읽기모드) 국가명.txt파일 불러와서
for file in files:
    with open(file, 'r') as f:
        #print(file)   
        #국가별.txt파일 안에 있는 이름들 한줄씩 가져와서 양끝 개행문자없애고 중간에 개행문자를 기준으로 나눠서 names리스트에 저장
        names = f.read().strip().split('\n')
        #혹시 개행문자가 두번(이름과 이름사이에 엔터두번친 경우) 있어서
        #빈문자열이 들어갈 수 있으니 빈 문자열을 제외한 리스트 생성
        names = [line for line in names if line != '']
        #print(names)
        
        # 파일명에서 국가명을 추출 (예: 'names/French.txt' -> 'French')
        #lang = file.split('/')[-1].split('.')[0]
        lang = os.path.splitext(os.path.basename(file))[0]
        print('lang',lang)

        # 국가명을 all_categories 리스트에 추가
        all_categories.append(lang)

        # 모든 이름을 소문자로 변환 (대소문자를 구분하지 않기 위해)
        names = [n.lower() for n in names]

        # 각 이름에서 알파벳이 아닌 문자를 제거 (예: 특수 문자나 숫자는 제거)
        #names = [''.join([c for c in n if c in all_letters]) for n in names]
        names = [''.join(filter(lambda c: c in all_letters, n)) for n in names]

        # 해당 국가의 이름들을 category_names 딕셔너리에 저장
        category_names[lang] = names
        #print(category_names)
        # 국가명과 그 국가의 이름 개수, 첫 번째부터 세 번째까지의 이름을 출력
        print(f'{lang}: {len(names)} |', names[0], names[1], names[2])

    # 총 국가 카테고리 수를 계산
    n_categories = len(all_categories)

##그전에, 이름데이터를 모두 같은 길이의 텐서로 입력받기위해서 가장 긴 이름의 길이로 맞추거나/PADDING을 준다.
##만약 관리와 가독성을 우선시한다면 반복문을 사용하는 것이 좋고, 성능을 더 중시한다면 리스트 컴프리헨션이나 NumPy를 사용하는 것이 좋습니다.
'''
#1)넘파이사용
import numpy as np
lengths = [len(name) for names_list in category_names.values() for name in names_list]
max_length = np.max(lengths)

#2)판다스사용
import pandas as pd
# 모든 이름을 리스트로 변환했다가 시리즈로 데이터타입변환하고 max를구해야함.
all_names = [name for names_list in category_names.values() for name in names_list]
name_lengths = pd.Series(all_names).apply(len)
max_length = name_lengths.max()

#3)리스트컴프리헨션사용
max_length = max(len(name) for names_list in category_names.values() for name in names_list)

#4)반복문사용
max_length = 0
for names_list in category_names.values():
    for name in names_list:
        max_length = max(max_length, len(name))
'''
max_length = max(len(name) for names_list in category_names.values() for name in names_list)

##이름과 국가명 데이터를 딥러닝 하기위해 숫자텐서로 바꾼다.
#이름을 텐서로 변환하는 함수 정의
# def name_to_tensor(name):
#     #패딩으로 최대글자수 길이만큼의 텐서만들기 
#     padding_tensor = torch.zeros(max_length, dtype=torch.long)
#     # 이름의 각 문자를 텐서에 넣기
#     for i, letter in enumerate(name):
#         padding_tensor[i] = ord(letter)  # 문자 유니코드 값 할당
#     print(padding_tensor)
#     # 결과 출력
#     return padding_tensor


def name_to_tensor(name):
    # 패딩으로 최대글자수 길이만큼의 텐서 만들기 
    padding_tensor = torch.zeros(max_length, dtype=torch.long)
    # 이름의 각 문자를 텐서에 넣기
    for i, letter in enumerate(name):
        if letter in letter_to_index:  # 알파벳에 있는 경우만
            padding_tensor[i] = letter_to_index[letter]  # 인덱스 값 할당
    return padding_tensor


# 임베딩 크기 정의
embedding_dim = 10  # 임베딩 차원 (나중에 변화시켜보기,하이퍼파라미터로)
# 임베딩 레이어 정의
embedding = nn.Embedding(num_embeddings = num_letters, embedding_dim = embedding_dim)
# 임베딩 레이어 생성
#embedding = nn.Embedding(num_embeddings=max_length, embedding_dim=embedding_dim, padding_idx=0)
# 임베딩 레이어 생성 (올바른 방식)
#embedding = nn.Embedding(num_embeddings=num_letters, embedding_dim=embedding_dim)


# 임베딩을 수행하는 함수
def name_to_embedding(name):
    # 이름을 텐서로 변환
    name_tensor = name_to_tensor(name)
    # 임베딩 수행
    embedded_tensor = embedding(name_tensor)
    return embedded_tensor


# 예시 이름
name = "alice"
# 임베딩된 텐서 얻기
embedded_name = name_to_embedding(name)
# 결과 출력
print("Embedded tensor name:", embedded_name)
print('-------------')

# # 모든 이름을 텐서로 변환
# tensors = {lang: torch.stack([name_to_tensor(name) for name in names]) for lang, names in category_names.items()}



##국가명데이터는 정답으로 비교할거니까 인덱스형태로 변환한다.
# #국가명을 인덱스로 변환하는 함수 정의
# 국가명을 인덱스로 매핑할 딕셔너리 생성
# category_to_index = {category: i for i, category in enumerate(all_categories)}

# # 국가명을 인덱스로 변환하는 함수
# def category_to_tensor(category):
#     index = category_to_index[category]
#     return torch.tensor(index, dtype=torch.long)

# # 테스트 예시
# category_name = "Italian"
# category_tensor = category_to_tensor(category_name)
# print(f"Category '{category_name}' as tensor:", category_tensor)



 

# import torch

# # 문자 인덱스 매핑 사전 만들기
# letter_to_index = {letter: i for i, letter in enumerate(all_letters)}
# index_to_letter = {i: letter for i, letter in enumerate(all_letters)}

# # 최대 이름 길이 정의
# max_length = max(len(name) for names_list in category_names.values() for name in names_list)

# # 이름을 텐서로 변환하는 함수 정의
# def name_to_tensor(name):
#     # 이름을 문자 인덱스로 변환하고, 패딩을 추가하여 고정 길이로 맞춤
#     tensor = torch.zeros(max_length, n_letters)  # (max_length, n_letters) 크기의 텐서
#     for idx, letter in enumerate(name):
#         tensor[idx][letter_to_index[letter]] = 1  # 원-핫 인코딩
#     return tensor

# # 국가 레이블을 인덱스로 변환하는 함수 정의
# def category_to_tensor(category):
#     return torch.tensor(all_categories.index(category))

# # 예시: 첫 번째 국가의 이름을 텐서로 변환
# example_name = category_names[all_categories[0]][0]  # 첫 번째 국가의 첫 번째 이름
# tensor_name = name_to_tensor(example_name)
# print(f'Name: {example_name} -> Tensor Shape: {tensor_name.shape}')





#데이터셋 나누기








#2. 데이터모델만들기 -> model.py

#3. 데이터 학습시키기 -> model.py

#4. 데이터 평가하기 -> main.py

#5. 하이퍼파라미터 튜닝 -> model.py
'''
1.embedding(input_sequence,dim) 임베딩 차원수 바꿔보기
2.패딩 0위치  좌측정렬/우측정렬/가운데정렬 바꿔보기 (피드포워드에서는 상관없다함.)
3.패딩을 0말고 1이나 2로 해보기
4.알파벳데이터->벡터변환할때 값을 ord()써서 유니코드로 했을때 97~122랑/ 1~26으로 했을때/ 알파벳한개씩 원핫인코딩했을때 비교해보기
5.

'''
