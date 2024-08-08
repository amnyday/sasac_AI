class MyDate:

    def __init__(self, year = 0, month = 0, day = 0, hour = 0, minute = 0, sec = 0):
        pass 

    def __add__(self, other):
        pass 

    def __sub__(self, other):
        pass 

    def __eq__(self, other):
        pass 

    def __lt__(self, other):
        pass 
    
    def __le__(self, other):
        pass 

    def __gt__(self, other):
        pass 

    def __ge__(self, other):
        pass 

if __name__ == '__main__':
    d0 = MyDate()
    d1 = MyDate(2022, 4, 1, 14, 30)
    d2 = MyDate(2024, 8, 100, 23, 10) # should raise an error 
    d3 = MyDate(2024, 2, 30)

    d3 = MyDate(day = 1) 
    assert d1 + d3 == MyDate(2022, 4, 2, 14, 30)
    assert d1 - d3 == MyDate(2022, 3, 31, 14, 30) 

    assert d1 < d2 


    
    
