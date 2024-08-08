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
    def gubun_leaf(target_year):
        if target_year%4 == 0 and target_year%100 > 0 or target_year%400 ==0: #시작날짜가 윤년이면서 2월29일 이전이면, 더한날짜에 -1일
            leaf = 'True'
            print('leaf이다')
        else: #시작날짜가 평년이면
            leaf = 'False'
            print('basic이다')        
        return leaf
        
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
           
        self.gubun_leaf(self.year)               
        add_days = 0
        # if leaf_year == 'True':
        #     m = leaf_m
        # else:
        #     m = basic_m
        
        def get_add_days(leaf,target_year,target_month,target_day): #추가된 일수 구할때 쓰는함수
            #몇월몇일이면, 구하는법       윤년이면 2월 일수-1

            add_days = 0
            if target_year%4 == 0 and target_year%100 > 0 or target_year%400 ==0: #시작날짜가 윤년이면서 2월29일 이전이면, 더한날짜에 -1일
                m = leaf_m #더한날짜구하는거는 2024년1월에 더하는거면 2024년1월부터 몇일지난거에 더할건지 초기일수구해놓고
                #더하는날짜의 연도가 윤년인지평년인지구해서 그날짜일수 더해서 다시 일자계산
                leaf = 'True'
                print('leaf이다')
            else: #시작날짜가 평년이면
                m = basic_m
                leaf = 'False'
                print('basic이다')
                
            for i, days in enumerate(m):  #i = 0,1,2,3,4,5,~
                #월에맞는 날짜수를 가져온다. mo = 
                print(i,days) 
                if  target_month == i+1 :
                    break
                else:
                    add_days += days #해당월 전달까지의 날짜수를 각각 곱해서 더해놓는다.
                    #1*31 2*28 #3월 더하는거면 2월까지의 날짜를 더해둔다.
            add_days += target_day #총 더해지는날짜
            print(f'{target_month}월 까지의 날짜',add_days)
            
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
        start_day = get_add_days(leaf,self.year, self.month, self.day) 
        end_day = get_add_days(leaf,other.year,other.month, other.day)
        
        print('start_day,end_day:',start_day,end_day)
        
        
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

    def __gt__(self, other):# d1이 d2보다 클때 d1 > d2
        pass 

    def __ge__(self, other):# d1가 d2보다 같거나 클때 d1 >= d2
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
    MyDate(year= 2024, month = 12, day = 31 ) + MyDate(month =12, day =31)
    
    
