#7/18 실습
#엔터로스프릿해서 쪼개서쓰기
#만약 text 마지막 줄이 = (assignment)가 아니면, 계산값 리턴 맞으면 Name리턴, 만약 게산불가하면 NameError리턴
#인덴트는 다 없다고 가정해라. 
import re
simple_python = '''
a =2
b =1
a+b+3
'''

def eval_simple_python(text):
    lines = text.split('\n')
    print(lines)

    line = lines[-1] #마지막줄의 문자열 받아옴
    if re.fullmatch(r'([^=]+)=', line) : # 정규식표현으로 =기호가 1개들어있는지확인 #assignment면 
        res = match.group(1) # Name리턴
        return res
        
    else: #assignment가 아니면 계산값 리턴 
        if : #계산값 리턴
           res = '계산값'
           return res
        else: #계산이불가하면 에러리턴
           raise NameError("Name cannot be deception")         
    
eval_simple_python(simple_python)
