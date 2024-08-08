all_smallcase_letters = 'abcdefghijklmnopqrstuvwxyz'

# --------------------------------------------
# 1. list/tuple 기초 예제들 
# 
# a는 1,2,3이 들어간 튜플, 
# b는 a부터 z까지 모든 알파벳 소문자가 들어간 리스트가 되도록 만들어보세요. 
# b를 만들 때 위에 주어진 all_smallcase_letters와 for loop를 사용해도 좋고, 손으로 다 쳐도 좋습니다. 
# --------------------------------------------

# write your code here 
a = (1,2,3)
b = []

for s in all_smallcase_letters:
    b.append(s)
print(b)

# --------------------------------------------
# 2. dict 기초 예제 
# 
# 1) upper_to_lower
#
# upper_to_lower은 모든 대문자 알파벳(ex. A)을 key로 가지고, 대응하는 소문자 알파벳(ex. a)을 value로 가지는 dict입니다. 
# 위에서 만든 b와 for loop를 이용해서 upper_to_lower을 만들어보세요. 
# 
# 2) lower_to_upper
#
# upper_to_lower과 반대로 된 dict를 만들어보세요. 
# 
# 3) alpha_to_number
# 
# 소문자 알파벳 각각을 key, 몇 번째 알파벳인지를 value로 가지는 dict를 만들어보세요. 
# 위 all_smallcase_letters와 enumerate함수를 사용하세요. 
# 알파벳 순서는 1부터 시작합니다. ex) alpha_to_number['a'] = 1
# 
# 4) number_to_alphabet 
#
# 1부터 26까지의 수를 key로, 소문자, 대문자로 이뤄진 문자열 2개의 튜플을 value로 가지는 dict를 만들어보세요. 
# --------------------------------------------

# write your code here 
#1)
upper_to_lower = {}
for s in all_smallcase_letters:
    #upper_to_lower = {s.upper():s.lower()}
    upper_to_lower[s.upper()] = s.lower()
print(1)
print(upper_to_lower)
#2)
lower_to_upper = {}
for s in all_smallcase_letters:
    lower_to_upper[s.lower()] = s.upper()
print(2)
print(lower_to_upper)
#3)
alpha_to_number = {}
i = 0
for s in enumerate(all_smallcase_letters):
    i += 1
    alpha_to_number[s[1]] = i
print(3)
print(alpha_to_number)
#4)
number_to_alphabet = {}
i = 0
for s in all_smallcase_letters:
    i += 1
    number_to_alphabet[i] = (s.lower(),s.upper())
print(4)
print(number_to_alphabet)
# --------------------------------------------
# 3. 주어진 문자열의 대소문자 바꾸기 
#
# 위 2에서 만든 dict들을 이용하여, 아래 문제들을 풀어보세요. 
#  
# 1) 주어진 문자열을 모두 대문자로 바꾼 문자열을 만들어보세요. 
# 2) 주어진 문자열을 모두 소문자로 바꾼 문자열을 만들어보세요. 
# 3) 주어진 문자열에서 대문자는 모두 소문자로, 소문자는 모두 대문자로 바꾼 문자열을 만들어보세요. 
# 4) 주어진 문자열이 모두 알파벳이면 True, 아니면 False를 출력하는 코드를 짜보세요. 
# --------------------------------------------

'''
a = 'absdf123SAFDSDF'

#1)
input_str = input('1번문제 문자열 작성(전부 소문자로):')
result_str = ''
for s in input_str:
     result_str += lower_to_upper[s]
print('result_str:',result_str)
 
#2)
input_str = input('2번문제 문자열 작성(전부 대문자로):')
result_str = ''
for s in input_str:
    result_str += upper_to_lower[s]
print('result_str:',result_str)

#3)
input_str = input('3번문제 문자열 작성(대소문자숫자섞어서):')
result_str = ''

#for s in input_str:
#    if s.islower():
#        result_str += lower_to_upper[s]
#    elif s.isupper():
#        result_str += upper_to_lower[s]
#    else:
#        result_str += s
#print('result_str:',result_str)

def lower_or_upper(s):
    if s.islower():
        return lower_to_upper[s]
    elif s.isupper():
        return upper_to_lower[s]
    else:
        return s
    
for s in input_str:
    result_str += lower_or_upper(s)
print(result_str)

#4)
input_str = input('4번문제 문자열 작성(모두 알파벳인지 확인):')
def all_alpha(input_str):
    gubun = ''
    for s in input_str:
        if s.isalpha() == False:
            gubun = 'False'
            break
    if gubun == 'False':
        print(gubun)
    else:   
        print('True')    

all_alpha(input_str)       
    
'''   
# --------------------------------------------
# 4. 다양한 패턴 찍어보기 
# 
# 1) 피라미드 찍어보기 - 1 
#
# 다음 패턴을 프린트해보세요. 
#
#     *
#    ***
#   *****
#  *******
# *********
# --------------------------------------------

# write your code here
# #빈칸 4 3 2 1 0 양옆
'''
n = 5
k = 5
j = 0
for i in range(n):
    k -= 1
    print(' '*k+'*'*(n-k)+'*'*j+' '*k,end='')
    print()
    j += 1
''' 
# --------------------------------------------
# 2) 피라미드 찍어보기 - 2 
# 
# 다음 패턴을 프린트해보세요. 
# 
#     * 
#    * * 
#   * * * 
#  * * * * 
# * * * * * 
# --------------------------------------------

# write your code here 
'''
n = 5
k = 5
for i in range(n):
    k -= 1
    print(' '*k+'* '*(n-k)+' '*k,end='')
    print()
'''
# --------------------------------------------
# 3) 피라미드 찍어보기 - 3 
# 
# 다음 패턴을 프린트해보세요. 
#
#     A 
#    A B 
#   A B C 
#  A B C D 
# A B C D E 
# --------------------------------------------

# write your code here 
'''
n = 5
k = 5
str_list = ['A','B','C','D','E']
str = 'A '
'''
# for i in range(n):
#     k -= 1
#     print(' '*k+(str_list[n-k-1]+' ')*(n-k)+' '*k,end='')
#     print()

# for i in range(n): #5
#     k -= 1
#     for j in range(n-k):
#         print(' '*k+'* '*(n-k)+' '*k,end='')
#     print()
'''
for i in range(n):
    k -= 1
    print(' '*k,end='')
    for i in range(n-k):
        str = str_list[i] + ' '
        print(str,end='')
    print(' '*k,end='')
    print()
'''
# --------------------------------------------
# 4) 피라미드 찍어보기 - 4 
# 
# 다음 패턴을 프린트해보세요. 
# 
#       1 
#      1 1 
#     1 2 1 
#    1 3 3 1 
#   1 4 6 4 1
# --------------------------------------------

# write your code here 
# 0 1 2 3 4 아니고 파스칼의 삼각형
'''
n = 5
k = 5
for i in range(n):
    k -= 1
    print(' '*k,end='')
    for i in range(n-k):
        str = str(i) + ' '
        print(str,end='')
    print(' '*k,end='')
    print()
'''

'''
n = 0

number_list = [1]
temp_list = []
temp_int = 1

for i in range(0,n) :
    if n == 0 :
        print(number_list)
    elif n == 1 :
        number_list.append(1) #[1,1]
        print(number_list)
    elif n >= 2 :
        number_list = [1,1]
        for k in range(n-1): #또는 i-1?
            print('k:',k)
            temp_list.append(number_list[k]+number_list[k+1])
            
            #temp_list[k-1] = (number_list[k-2] + number_list[k-1]) # n=2-> number_list[0] + number_list[1] 1+1=2
            #temp_list.append(temp_int)
            
            #number_list.append(temp_int)
            #number_list.append(1)
            #temp_list[k] = 1
        number_list.append(temp_list)
        #number_list.append(temp_list)
        print('temp:',temp_list)
        print('number_list:',number_list)
    
'''

'''
#신승우강사님
def pascal(n):
    def generate_next_line(last_line):
        n = len(last_line) + 1
        
        next_line = [last_line[0]]

        for i in range(n-2):
            next_line.append(last_line[i] + last_line[i+1])

        next_line.append(last_line[n-2])

        return next_line
    
    lines = [[1], [1,1]]
    
    while len(lines) != n:
        lines.append(generate_next_line(lines[-1]))

    space = ' '

    def fill(number, digits, fill_with = '0'):
        number_digit = get_digit(number)
        return (digits - number_digit) * fill_with + str(number)

    def get_digit(number):
        # int(log_10(n))
        # 123 
        digit = 1
        
        while True:
            if number < 10:
                break 
            else:
                digit += 1 
                number = number // 10 
        
        return digit 

    # print(fill(123, 4)) # 0123
    # print(fill(123, 5)) # 00123
    # print(fill(123, 4, '|')) # |123

    max_number = max(lines[-1])
    max_digit = get_digit(max_number)

    space = ' ' * max_digit

    for idx, line in enumerate(lines):
        print((n-1-idx)*space + space.join([\
            fill(e, max_digit, ' ') for e in line]))
    
    return lines 

for line in pascal(12):
    print(line)
    
'''   

'''챗gpt --리스트로 찍기
def pascal_triangle(n):
    if n < 0:
        return "Invalid input: n must be a non-negative integer."
    
    number_list = [[1]]  # Initialize with the first row of Pascal's triangle
    
    for i in range(1, n + 1):
        temp_list = [1]  # Start with the first 1
        for j in range(1, i):
            # Append the sum of the two numbers above this position
            temp_list.append(number_list[i-1][j-1] + number_list[i-1][j])
        temp_list.append(1)  # End with the last 1
        number_list.append(temp_list)
    
    return number_list

# Example usage
n = int(input("Enter a non-negative integer: "))
triangle = pascal_triangle(n)
for line in triangle:
    print(line)
'''

'''챗gpt한테 내가짠거수정시킨거
def pascal_triangle(n):
    if n < 0:
        return "Invalid input: n must be a non-negative integer."

    if n == 0:
        print([1])
        return
    
    number_list = [1]
    print(number_list)  # Print the first row

    if n == 1:
        number_list.append(1)  # [1, 1]
        print(number_list)
        return

    for i in range(2, n + 1):
        temp_list = []
        number_list = [1] + number_list + [1]  # Create new row with leading and trailing 1s
        
        for k in range(len(number_list) - 1):
            temp_list.append(number_list[k] + number_list[k + 1])
        
        number_list = temp_list  # Update number_list to the new row
        print(number_list)  # Print the current row

# Example usage
n = int(input("Enter a non-negative integer: "))
pascal_triangle(n)
'''

'''#챗지피티한테 내가짠거수정시킨거 2 가운데정렬
def pascal_triangle(n):
    if n < 0:
        return "Invalid input: n must be a non-negative integer."
    
    number_list = [[1]]  # Initialize with the first row of Pascal's triangle
    
    if n == 0:
        print("1")
        return
    
    for i in range(1, n + 1):
        temp_list = [1]  # Start each row with a 1
        for j in range(1, i):
            temp_list.append(number_list[i-1][j-1] + number_list[i-1][j])
        temp_list.append(1)  # End each row with a 1
        number_list.append(temp_list)
    
    # Determine the width of the last row when joined with spaces
    max_width = len("   ".join(map(str, number_list[-1])))
    
    for row in number_list:
        row_str = "   ".join(map(str, row))
        print(row_str.center(max_width))

# Example usage
n = int(input("Enter a non-negative integer: "))
pascal_triangle(n)
'''


# for i in range(0,n):
#     if i == 0 :
#         print(number_list)
#     elif i == 1:
#         number_list.append(i)
#         print(number_list)
#     else: #2이상일때
#         number_list.append(number_list[0])
#         for j in range(n):
#             temp_list.append(number_list[n-2] + number_list[n-1])
#         number_list.append(temp_list)
#         number_list.append(number_list[n])
# print(number_list)    

# def pyramid4(n):
#     number_list = []
#     temp = []

#     for i in range(n): 
#         number_list.append(1)
#         temp.append(1)

#         if i < 2:
#             pass
#         else: 
#             for j in range(1,len(number_list)-1):
#                 temp[j] = number_list[j-1]+number_list[j]
#         print(blank*((n-1)-i), end="")

#         for j in range(len(number_list)):
#                 number_list[j] = temp[j]
#                 print(str(number_list[j]), blank, end ="")
#         print()

# pyramid4(how_many_star)
# --------------------------------------------
# 5) 다음 패턴을 찍어보세요. 
# 
# *         *         * 
#   *       *       *   
#     *     *     *     
#       *   *   *       
#         * * *         
#           *           
#         * * *         
#       *   *   *       
#     *     *     *     
#   *       *       *   
# *         *         *
# --------------------------------------------

# write your code here 
'''#신승우강사님 코드
star = '* '
space = '  '
#*한개+아홉개빈칸 *한개 아홉개빈칸+*한개

# top = [\
#     star + space*n + star + space*n +star,
#     space + star + space*(n-1) +star + space*(n-1) + star + space,
#     space*2 + star + space*(n-2) + star + space*(n-2) + star + space*2,
#     ....
#     space * n + star + space*0 +star + space*0 +star + space*n,]

n = 4
top = []
for i in range(n+1):
    top.append(space*i + star + space * (n-i)
               + star + space*(n-i) +star + space*i)
    
mid = [space*(n+1) + star +space*(n+1),]
bottom = []

for e in top:
    bottom = [e] +bottom

print('\n'.join(top+mid+bottom))  
'''

'''#내가짜는중코드    
n = 5
k = 5
for i in range(n):
    print('*' + ' '*9 + '*' + ' '*9 + '*')
    print(' '*2 + '*' + '*' + ' ')
    k -= 1
    print(' '*k+'* '*(n-k)+' '*k,end='')
    print()
'''

#내가짜는중코드2
for i in range(14):
    for j in range(14):
        if i == j or i+j == 14 :
            print('*',end='')
        else:
            print('  ',end='')
    print('')