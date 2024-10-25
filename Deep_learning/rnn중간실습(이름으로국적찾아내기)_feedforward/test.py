import torch
import torch.nn as nn
# def name_to_tensor(name, max_length):
#     # max_length 크기의 텐서 생성 (초기값은 0)
#     tensor = torch.zeros(max_length, dtype=torch.long)

#     for i, letter in enumerate(name):
#         if i < max_length:  # max_length보다 작은 경우에만
#             tensor[i] = ord(letter) - ord('a')  # 문자를 숫자로 변환

#     return tensor

# # 예시 사용
# name1 = "alex"     # 길이가 4인 이름
# name2 = "john"     # 길이가 4인 이름
# name3 = "elizabeth" # 길이가 9인 이름

# max_length = 10
# tensor1 = name_to_tensor(name1, max_length)  # [0, 11, 4, 23, 0, 0, 0, 0, 0, 0]
# tensor2 = name_to_tensor(name2, max_length)  # [9, 14, 7, 13, 0, 0, 0, 0, 0, 0]
# tensor3 = name_to_tensor(name3, max_length)  # [4, 11, 8, 25, 1, 5, 20, 8, 19, 0]

# print(tensor1)
# print(tensor2)
# print(tensor3)

max_length = 10
all_letters = 26
# 알파벳을 인덱스로 매핑 (예: a=0, b=1, ..., z=25)
letter_to_index = {letter: i for i, letter in enumerate(all_letters)}

# 이름을 인덱스 텐서로 변환하고 임베딩을 수행하는 함수
def name_to_embedding(name):
    # 최대 길이만큼의 텐서 초기화 (패딩 포함)
    name_tensor = torch.zeros(max_length, dtype=torch.long)
    # 알파벳 인덱스로 변환하여 텐서에 저장
    for i, letter in enumerate(name):
        name_tensor[i] = letter_to_index.get(letter, 0)  # 없는 문자면 0 할당
    # 임베딩 수행
    embedded_tensor = nn.embedding(name_tensor)
    return embedded_tensor

# 테스트
name = "alice"
embedded_name = name_to_embedding(name)
print("Embedded tensor name:", embedded_name)