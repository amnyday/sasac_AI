#[1] 피드포워드(MLP) 사용해서 이름의 국적을 추론하는 뉴럴넷 만들기
#1. 데이터셋만들기, 데이터전처리
import glob
import torch
from collections import defaultdict


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
n_letters = len(all_letters)
#알파벳 갯수 맞는지 확인
assert n_letters == 26

#이름가져온 데이터셋, 국적가져온 데이터셋 만들기
#names 파일 열어서(읽기모드) 국가명.txt파일 불러와서
for file in files:
    with open(file, 'r') as f:
        print(file)   
        #국가별.txt파일 안에 있는 이름들 한줄씩 가져와서 양끝 개행문자없애고 중간에 개행문자를 기준으로 나눠서 names리스트에 저장
        names = f.read().strip().split('\n')
        #혹시 개행문자가 두번(이름과 이름사이에 엔터두번친 경우) 있어서
        #빈문자열이 들어갈 수 있으니 빈 문자열을 제외한 리스트 생성
        names = [line for line in names if line != '']
        #print(names)
        #file
        lang = file.split('\\')[-1].split('.')[0]

    
    #print(lang)
    # : split으로 분리한 다음에 -1번째 0번째
    # all_categories.append(lang)

    
    # n
    
    
    














#2. 데이터모델만들기 -> model.py

#3. 데이터 학습시키기 -> model.py

#4. 데이터 평가하기 -> main.py

#5. 하이퍼파라미터 튜닝 -> model.py