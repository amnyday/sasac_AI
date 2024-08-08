class MyDate:

    #윤년 평년의 기준은  특정 년도를 4로 나눴을때 나눠 떨어지고 100으로 나눠 떨어지지 않거나 400으로 나눠 떨어지면 윤년,
    #아닌경우를 평년을 정의한다. 
    #2월의 마지막 날은 다르니 윤년일 경우에 마지막날을 29일, 평년일때는 28일로 하는 식도 적어준다.
    #2024 2025 2026 2027 2028
    '''윤 평 평 평 윤
    2024.1+3 = 2027 평 윤년이몇번인가? 2024 한번(-1일)
    2024.2.29+3 = 2027 평 윤년이몇번인가? 없음 
    2024.3+3 = 2027 평 윤년이몇번인가? 없음
    2024.1+4 = 2028.1 윤 윤년이몇번인가? 2024 한번(-1일)
    2024.1+4 = 2028.3 윤 윤년이몇번인가? 2024, 2028 두번 (-2일)
    2024.2.29+4 = 2028.1 윤 윤년이몇번인가? 없음
    2024.2.29+4 = 2028.3 윤 윤년이몇번인가? 2028 한번(-1일)
    2024.3+4 = 2028.1 윤 윤년이몇번인가? 없음
    2024.3+4 = 2028.3 윤 윤년이몇번인가? 2028 한번(-1일)
    
   
    if year%4 == 0 and year%100 > 0 or year%400 ==0: #시작날짜가 윤년이면서 2월29일 이전이면, 더한날짜에 -1일
        leaf_year = true  ##더한결과 년도가 평년이면, 그냥
        ##더한결과 년도가 윤년이고, 2월29일 
        #시작날짜가 윤년이면서 2월29일 이후이면 그냥날짜더하기
        #시작날짜가 윤년이면서 더한결과년도가 
    
        #예를들어 2024년 2월29일 + 3월2일 =>? 더하는 날짜를 일수로계산해야할듯
        m = [31,29,31,30,31,30,31,31,30,31,30,31]
        for i in m:
    '''        
            
    
    def __init__(self, year = 0, month = 0, day = 0, hour = 0, minute = 0, sec = 0):  
        self.name = self
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.sec = sec
            
        #달의 마지막날짜
        #m = [31,29,31,30,31,30,31,31,30,31,30,31]
        #m_30 = [4,6,9,11]
        #m_31 = [1,3,5,7,8,10,12]
    
    '''
        if month > 12:
            raise ValueError ('month > 12 error')#assert 'month > 12 error'
        elif day > 32:
            raise ValueError ('day > 32 error') #assert 'day > 32 error'
        elif month in m_30:
            if day > 30:
                raise ValueError('day is not 30!')
        elif month in m_31:
            if day > 31:
                raise ValueError('day is not 31!')
        elif month == 2:
            if year%4 == 0 and year%100 > 0 or year%400 ==0:
                if day >29:
                    raise ValueError('윤년은 29일까지임')
            else:
                if day >28:
                    raise ValueError("평년은 28일까지임")            
        elif hour > 60 :
            raise ValueError('hour > error') #assert 'hour > 60 error'    
        elif minute > 60 :
            raise ValueError('minute > 60 error') #assert 'minute > 60 error'
        elif sec >60 : 
            raise ValueError('sec > 60 error') #assert 'sec > 60 error'
            
        #print(f'{year}/{month}/{day}.{hour}:{minute}:{sec}')
        #self = f'{year}/{month}/{day}.{hour}:{minute}:{sec}'
    '''
    #윤년확인 함수 - 윤년이면 'True'/평년이면 'False'
    def gubun_leaf(self,target_year):
        basic_m = [31,28,31,30,31,30,31,31,30,31,30,31]
        leaf_m = [31,29,31,30,31,30,31,31,30,31,30,31]
        
        if target_year%4 == 0 and target_year%100 > 0 or target_year%400 ==0: #시작날짜가 윤년이면서 2월29일 이전이면, 더한날짜에 -1일
            #leaf = 'True'
            print('leaf이다')
            return leaf_m
        else: #시작날짜가 평년이면
            #leaf = 'False'
            print('basic이다')        
            return basic_m
                
    def __add__(self, other): #더할 때 쓰는 애 
        f'{self.year}/{self.month}/{self.day}.{self.hour}:{self.minute}:{self.sec}'
        year = self.year + other.year
        if (self.month + other.month) //2 > 0 :
            year += (self.month + other.month) //2 #연도에추가
            month = (self.month + other.month) % 2 #나머지
        else:
            month = self.month + other.month        
               
        basic_m = [31,28,31,30,31,30,31,31,30,31,30,31]
        leaf_m = [31,29,31,30,31,30,31,31,30,31,30,31]

        #추가된 일수 구한다.  몇월몇일이면, 구하는법 윤년이면 2월 일수-1
        start_day = 0 #a
        add_days = 0 #b
            
        #시작년도 윤년/평년 계산해서 시작일수 구하기 a
        # if self.gubun_leaf(self.year): #윤년이면 366
        #     m = leaf_m
        # else: #평년이면
        #     m = basic_m
            
        #시작년도 윤년/평년 계산해서 시작일수 구하기 a   
        for i, days in enumerate(self.gubun_leaf(self.year)):  #i = 0,1,2,3,잘나옴. 
            #월에맞는 날짜수를 가져온다.
            print(i,days) 
            if  self.month == i+1:  #3==3: 1==1:
                start_day += self.day #일수를 더한다. #없으면 0
                break;
            else:
                start_day += days #해당월 전달까지의 날짜수를 각각 곱해서 더해놓는다.
        print('start_day',start_day)
        
        #시작일수 a + b 추가월,일을 일수로 바꾼것을 더한다. 년도는 년도더한걸로 평년윤년을 구한다.
        for i, days in enumerate(self.gubun_leaf(self.year + other.year)): #더한 년도가 윤년이면 2024+3 =2027년도에 일수로 더한다.
            print(i,days) 
            if  other.month == i+1:  #3==3: 1==1:
                add_days += other.day #일수를 더한다. #없으면 0
                break;
            else:
                add_days += days #해당월 전달까지의 날짜수를 각각 곱해서 더해놓는다.
        print('add_days',add_days)
        
        add_day = start_day + add_days 
        print('현재 add_day',add_day)
        if self.gubun_leaf(add_day) == basic_m: #평년이면 
            if add_day > 365: #일수합친게 일년일수 넘으면 year+1
                year = self.year + other.year + 1 #add_day//365
                #add_day #더한년도의 남은날->월,일로 나눠야함.
        
        else: #윤년이면
            if add_day > 366: #일수합친게 일년일수 넘으면 year+1
                self.year + other.year + 1
        
        #self.gubun_leaf(self.year + other.year + 1) self.gubun_leaf(self.year)
        #시작년도에 년수 더한 year가 시작년도가 윤년이면 다시 구분 그시작년도 1년일수366<a+b넘으면 year+1
        #추가년도 윤년/평년 계산해서 추가일수만 구하기 b  
        
        print('start_day,add_days:',start_day,add_days)
                    
        #여기다가 해당월날짜도 더해준다.  예를들어 2024년 2월29일 + 3월2일 =>? 더하는 날짜를 일수로계산해서
        
        #2024년12월31일 +12월31일 =>?최대2026년1월1일?
        #변수초기화
        hour = 0
        minute = 0
        sec = 0
        day = 0

        start_day = 0
        end_day = 0
        
        sec = self.sec + other.sec
        minute += (sec//60 + self.minute + other.minute)
        sec = sec%60
        
        hour += (minute//60 + self.hour + other.hour)
        minute = minute%60
        
        day += (hour//24 + self.day + add_days)  
        hour = hour%24

        
        #윤년계산하지말고 일단 날짜만더하면
        #시작날짜를 일수계산해서  + add_days추가된일수 더한다음 
        
        #start_day = get_add_days(leaf,self.year, self.month, self.day) 
        #end_day = get_add_days(leaf,other.year,other.month, other.day)
    
        
        
        print(f'{year}/{month}/{day}.{hour}:{minute}:{sec}')
        
    def __sub__(self, other):# 뺄 때 쓰는 애 
        # My_date.__sub__(d1,d2) 형식으로 부름 
        pass 

    def __eq__(self, other): # 같을 때 애들이 같은지 test
        pass 

    def __lt__(self, other):# d1, d2가 주어졌을 때, d1이 d2보다 작을 때 ex) d1 < d2
        pass 
    
    def __le__(self, other):# d1이 d2보다 같거나 작을 때 d1 <= d2
        pass 

    def __gt__(self, other):# d1이 d2보다 클때 true리턴 아니면 false리턴 d1 > d2 
        pass 

    def __ge__(self, other):# d1가 d2보다 같거나 클때 true리턴 아니면 false리턴d1 >= d2
        pass 

if __name__ == '__main__':
    d0 = MyDate()
    d1 = MyDate(2022, 4, 1, 14, 30)
    d2 = MyDate(2024, 8, 100, 23, 10) # should raise an error 
    d3 = MyDate(2024, 2, 30)

    d3 = MyDate(day = 1) 

    #assert d1 + d3 == MyDate(2022, 4, 2, 14, 30)
    #assert d1 - d3 == MyDate(2022, 3, 31, 14, 30) 

    #assert d1 < d2 

    #print(d1)
    #MyDate(year= 2024, month = 12, day = 31 ) + MyDate(month =12, day =31)
    MyDate(year= 2024, month = 2, day = 29 ) + MyDate(year = 3, month =3, day =2)    
    
