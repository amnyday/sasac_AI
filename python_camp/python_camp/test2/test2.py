# def c(lst):
#     print(lst[0])
#     return c(lst[1:]) 

# c(list(range(10)))



#예를들어 2024년 2월29일 + 12월2일 =>? 더하는 날짜를 일수로계산해야할듯
m = [31,28,31,30,31,30,31,31,30,31,30,31]

month = 3
add_days = 0
for i, days in enumerate(m):  #i = 0,1,2,3,4,5,~
    #월에맞는 날짜수를 가져온다. mo = 
    print(i,days)
    if  month == i+1 :
        break
    else:
        add_days += days #해당월 전달까지의 날짜수를 각각 곱해서 더해놓는다.
        #1*31 2*28 #3월 더하는거면 2월까지의 날짜를 더해둔다.
print(f'{month}월 까지의 날짜',add_days)

#여기다가 날짜를 더해준다.
add_days += 2
print(add_days)


from datetime import datetime

X = datetime(2024,12,31)  # 오늘 날짜를 저장
Y = datetime(2000,12,31)  # 2023년 5월 1일을 저장
#X+Y 에러