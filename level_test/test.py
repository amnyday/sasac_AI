# a = {1:2,3:4,4:5}

# print(a[1])
# print(a[1:3])

'''
all_smallcase_letters = 'abcdefghijklmnopqrstuvwxyz'
upper_to_lower = {}
lower_to_upper = {}
for c in all_smallcase_letters:
    upper_to_lower[c.upper()] = c
    lower_to_upper[c] = c.upper()
    
a = 'absdf123SAFDSDF'
ans_1 = ''
ans_2 = ''

for d in a:
    if c in upper_to_lower:
        ans_1 += upper_to_lower[c]
        ans_2 += c
    else:
        ans_1 += d
        ans_2 += lower_to_upper[c]
        
print('ans_1',ans_1)
print('ans_2',ans_2)        
'''

# def hello(hello_to):
#     print('hello '+hello_to)
# hello('world')
# print(hello('world'))

'''
import sys
#input = sys.stdin.readline

n = int(input('숫자입력:'))

stars = [[' ']*2*n for _ in range(n)]

def recursion(i, j, size):
    if size == 3:
        stars[i][j] = '*'
        stars[i + 1][j - 1] = stars[i + 1][j + 1] = "*"
        for k in range(-2, 3):
            stars[i + 2][j - k] = "*"
    
    else:
        newSize = size//2
        recursion(i, j, newSize)
        recursion(i + newSize, j - newSize, newSize)
        recursion(i + newSize, j + newSize, newSize)

recursion(0, n - 1, n)
for star in stars:
    print("".join(star))
'''

'''
n = int(input())

def star(l):
    if l == 3:
        return ['  *  ',' * * ','*****']

    arr = star(l//2)
    stars = []
    for i in arr:
        stars.append(' '*(l//2)+i+' '*(l//2))

    for i in arr:
        stars.append(i + ' ' + i)

    return stars

print('\n'.join(star(n)))'''


'''
#8/6 디렉토리경로확인 실습
import os
#import pickle

print(os.getcwd())

for elem in os.listdir():
    #print(elem, end = '')
    if os.path.isdir(elem):
        print(f'<DIR>\t\t{elem}')
    elif '.' in elem:
        extension = elem.split('.')[-1]
        print(f'{extension} file\t\t{elem}')
        
def create_dir(directory_name):
    if not os.path.exists(directory_name):
        print(f'{directory_name} does not exists;')
        os.makedirs(directory_name)
    else:
        print(f'{directory_name} does exists;')
create_dir('hello word')
'''

#8/6신승우강사님 실습
#2.file기초예제
#1)open이해하기
#2)파일읽기,써보기


'''#쓰기모드
#w+ 모드로 글수정하면, 이전글 전부삭제되고,    
#a+ 모드로 글수정하면, 이전글 냅두고 다음줄에추가
f = open('example2.txt', 'a', encoding = 'utf-8')

for i in range(100):
    #파일에글쓰는법1)print(i,file=f)
    #파일에글쓰는법2)f.write(str(i)+ '\n')
    #f.write(str('w'))
    print(i,file=f)
f.close()
'''

'''#읽기모드
#f.read()는 한번에 읽어온다.
#f.readlines()는 한줄씩 읽어온다. 이전에 읽었던위치다음부터
f = open('example.txt', 'r', encoding = 'utf-8')

print(f.readline())
print(f.readline())
for line in f.readlines():
    print(line)
    
f.close()
'''

#pickle 기초예제
import pickle
d = {}

pickle.dump(d, open('empty_dict.pickle', 'wb+'))

e = pickle.load(open('empty_dict.pickle', 'rb'))

print(e)













