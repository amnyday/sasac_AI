#파이썬 타입 선언
def EvenOddChecker(number:int) -> str:
    assert number > 0
    if number % 2 == 0:
        return "Even"
    else:
        return "Odd"
    
    
from typing import TypedDict, Optional

class Car(TypedDict, total=False):
    make: str
    model: str
    year: int
    color: Optional[str]  # 선택적 필드

car_info: Car = {
    'make': 'Toyota',
    'model': 'Corolla',
    'year': 2020
}

print(car_info)
