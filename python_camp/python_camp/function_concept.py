# --------------------------------------------
# 1. 함수의 다양한 입력들 살펴보기 
#
# 1) input이 없는 함수 
# 2) input이 여러 개 있는 함수 
# 3) input이 정해지지 않은 갯수만큼 있는 함수 
# --------------------------------------------

def pi():
    """원주율을 소숫점 두 자리까지 반환하는 함수
    """
    import math
    return round(math.pi,2)
    
def left_append(lst, elem):
    """lst의 왼쪽에 elem을 넣고, lst를 반환하는 함수 
    """
    temp_lst = lst
    result_lst = []
    result_lst.append(elem)
    result_lst = result_lst + temp_lst
    return result_lst

def left_extend(lst, *elem):
    """lst의 왼쪽에 정해지지 않은 갯수의 elem을 넣고 lst를 반환하는 함수 
    """
    temp_lst = lst
    print(elem) 
    result_lst = []
    result_lst.append(elem)
    result_lst = result_lst + temp_lst
    return result_lst
# --------------------------------------------
# 2. 함수의 call stack 알아보기 
# 
# 1) 아래 함수 b()를 실행할 때, 실행된 함수의 순서는?
# --------------------------------------------

def a():
    return pi()

def b():
    return a()

#내답->함수b()를 실행하면, 1)함수b()실행되고 2)함수a()실행되고 pi()함수실행된다.

# --------------------------------------------
# 2) 아래 함수 c()를 실행할 때, 실행된 함수의 순서와 각 함수의 input은? 
# --------------------------------------------

def c(lst):
    print(lst[0])
    return c(lst[1:]) 

c(list(range(10)))

#내답->함수c()실행하면, 1)함수c()실행되고, list(range(10))이 함수의input으로들어간다.
#2)print()함수실행되고, lst[0:9] (0)이 함수의 input으로 들어가서 출력된다.
#3)함수c()가 실행되고, lst[1:9] (1~9)이 함수의 input으로 들어간다.
#4)print()함수실행되고, lst[1]이 함수의 input으로 들어가서 출력된다.
#5)함수c()가 실행되고, lst[2:9] (2~9)이 함수의 input으로 들어간다.
#6)print()함수실행되고, lst[2] 함수의 input으로 들어가서 출력된다.
#7)함수c가 실행되고, lst[3:9] 함수의 input으로 들어간다.
#8)print()함수실행되고, lst[3] 함수의 input으로 들어가서 출력된다.
#9까지 출력하고나서 더이상출력할거없어서 에러난다.

#-----------------------------------
print(pi())
a=[1,2]
print(left_append(a, 3))
print(left_extend(a, (2,3,5,6),(3,4,5,9,8)))