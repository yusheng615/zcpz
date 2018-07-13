import math


class Dictcase(dict):
    def __init__(self, **kw):
        super(Dictcase, self).__init__(**kw)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(r"'Dict' object has no attribute '%s' " % key)

    def __setattr__(self, key, value):
        self[key] = value


def add(a, b):
    if a > b:
        return a + b
    elif a == b:
        return a
    else:
        return a - b
    

def subtraction(a, b):
    if a < b:
        return a - b
    elif a == b:
        return a
    else:
        return a + b


def func(a):
    a = a * 2
    return a


def is_prime(n):
    print('验证数字' + str(n) + '是否指数开始')
    if n < 1:
        return False
    for i in range(2, round(math.sqrt(n))):
        # 只需要判断到数的开平发即可
        if not n % i:
            print(n/i)
            return False
        return True


# 用于判断一个12位的数字, 最后一位是前11位除以7的余模
def is_barcode(n):
    print('验证数字' + n + '是否符合规则开始')
    if len(n) != 12:
        return False

    n_count = int(n[0: 11])
    end_number = int(n[-1])
    print(n_count, end_number)

    if end_number == n_count % 7:
        return True

    else:
        print(n_count % 7)

    return False


def custom_range( start, end = None, sep = 1):
    if start > 0 and end == 0:
        start,end = end,start
    lst = []
    val = start
    while val < end:
        lst.append(val)
        val += sep
    return lst


def prime_m2n(m, n):
    lst = []
    if m > 1 and n > 1:
        for i in range(m, n):
            if is_prime(i) is True:
                lst.append(i)
        else:
            return lst
    else:
        return False
