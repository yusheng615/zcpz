
# 1,2,3,4个数组合无重复的三位数
def three_num():
    sum = 0
    for i in range(1,5,1):
        for j in range(1,5,1):
            for k in range(1,5,1):
                if (i != j) & (j!=k):
                    sum +=1
    return sum

# print(three_num())   # 36


# 一个整数，它加上100后是一个完全平方，再加上168又是一个完全平安
def find_sqrt():
    for i  in range(100):
        import math
        if math.sqrt(i+100) - int(math.sqrt(i+100)) ==0:
            x = int(math.sqrt(i +100))
            if math.sqrt(i+268) - int(math.sqrt(i+268)) == 0:
                y = int(math.sqrt(i +268))
                return x, y

# print(find_sqrt())  # (11,17)

# 归并排序
def merge_sort(lists):
    if len(lists) <= 1:
       return lists
    num = len(lists) / 2
    left = merge_sort(lists[:num])
    right = merge_sort(lists[num:])
    i, j = 0, 0
    result = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:]
    result += right[j:]
    return result


# 判断两个正整数的最大公约数
def hcf(x, y):
    if x > y:
        smaller=y
        biger=x
    else:
        smaller=x
        biger=y
    if biger%smaller==0:
        result=smaller
    else:
        for i in range(1,(smaller + 1)/2):
            if(x % i == 0) and (y % i == 0):
                result = i
    return result


# 找出所有不同的string，按照从小到大的顺序呢排序
def sort_str(world_str):
    while 1:
        if world_str == '':
            return ''
        s = [each if each.isalpha() else ' ' for each in world_str ]
        s = ''.join(s)
        s = set(s.split())
        s = list(s)
        s.sort()
        return s


