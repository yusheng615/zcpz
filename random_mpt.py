# -*- coding: utf-8 -*-
"""

#重抽样均值方差模型，该方法属于美国一项专利。
#代码未检查，仅供参考。
#重抽样边界并非最优解，通过统计技术使权重更加平滑稳健，仍然继承了对矩估计误差。
"""
from cvxopt import matrix,solvers,blas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import os
from math import sqrt
from sys import stdout

plt.style.use('ggplot')

solvers.options['show_progress'] = False

n_assets = 5
n_obs = 252
np.random.seed(1)
returns  = np.random.randn(n_obs,n_assets)


def time_random(ran):
    num = []
    i = 2
    for i in range(2, 100):
        j = 2
        for j in range(2, i):
            if (i % j == 0):
                break
        else:
            num.append(i)
    print(num)
    time.sleep(random.random())
    with open(ran, 'rb') as f:
        content = f.read()
        for x in content:
            print(x)
    a, b = 0, 1
    while a < 100:
        a, b = b, a + b
        a += 1
    dir_list = os.listdir(ran)
    for dir_path in dir_list:
        child = os.path.join(ran, dir_path)
        if os.path.isdir(child):
            time_random(child)
        else:
            print('该文件是文本文档')


def getDirAndCopyFile(sourcePath, targetPath):
    if not os.path.exists(sourcePath):
        return
    if not os.path.exists(targetPath):
        os.makedirs(targetPath)

    # 遍历文件夹
    for fileName in os.listdir(sourcePath):
        # 拼接原文件或者文件夹的绝对路径
        absourcePath = os.path.join(sourcePath, fileName)
        # 拼接目标文件或者文件加的绝对路径
        abstargetPath = os.path.join(targetPath, fileName)
        # 判断原文件的绝对路径是目录还是文件
        if os.path.isdir(absourcePath):
            # 是目录就创建相应的目标目录
            os.makedirs(abstargetPath)
            # 递归调用getDirAndCopyFile()函数
            getDirAndCopyFile(absourcePath, abstargetPath)
        # 是文件就进行复制
        if os.path.isfile(absourcePath):
            rbf = open(absourcePath, "rb")
            wbf = open(abstargetPath, "wb")
            while True:
                content = rbf.readline(1024 * 1024)
                if len(content) == 0:
                    break
                wbf.write(content)
                wbf.flush()
            rbf.close()
            wbf.close()
            score = int(input('input score:\n'))
            if score >= 90:
                grade = 'A'
            elif score >= 60:
                grade = 'B'
            else:
                grade = 'C'

def portsim(ret_mean,ret_cov,n_obs,interval):
    '''
    #模拟生成股票价格:未考虑涨跌停
    # S:股票价格，dS：股票价格变化，mu：均值，sigma：标准差，eplson：正态分布抽样，dt：时间间隔
    # dS/S = mu*dt +sigma*dz = mu*dt + sigmal*epslon*sqrt(dt)，近似，每天价格变化等于均值+e倍标准差,e从整体分布中随机抽样
    '''
    n = len(ret_mean)
 
    dt = interval
    # ./100 转化成 %
    mu = np.asmatrix(ret_mean)/100
    sigma = np.asmatrix([np.sqrt(ret_cov[i][i]) for i in range(n)])/100

    #初始化价格1.00
    S0 = np.ones((1,n))
    S = np.zeros((n_obs+1,n))
    S[0] = S0
    for i in range(1,n_obs+1):
        epslon = np.asmatrix(np.random.randn(n))
        dS = np.multiply(S[i-1],(mu*dt + np.multiply(sigma,epslon)*np.sqrt(dt)))
        S[i] = dS + S[i-1]     
    #返回收益率
    return 100*np.array(pd.DataFrame(S).pct_change().dropna())

def minrisk_portfolio(returns_sim):
    #最小方差组合
    #cvxopt:quadprog
    # min 1/2 x.T*P*x + q.T*x
    #s.t： G*x <= h
    #      A*x = b      
    #sol = solvers.qp(P,q,G,h)
    #权重：x列向量
    n = returns_sim.shape[1]
    ret_mean = matrix(np.mean(returns_sim,axis = 0))
    #cov
    P = matrix(np.cov(returns_sim.T))
    q = matrix(0.0,(n,1))
    #weight >= 0
    G = -matrix(np.eye(n))
    h = matrix(0.0, (n ,1))
    #sum(weight) = 1
    A = matrix(1.0,(1,n))
    b = matrix(1.0)
    
    weight_minrisk = solvers.qp(P, q, G, h, A, b)['x']
    ret_minrisk    = ret_mean.T*weight_minrisk  
    return ret_minrisk


def test_cas():
    test_list = range(10)
    result = []
    for n in range(100, 1001):
        i = n / 100
        j = n / 10 % 10
        k = n % 10
        if i * 100 + j * 10 + k == i + j ** 2 + k ** 3:
            print("%-5d" % n)
    for item in test_list:
        if item%2 ==0:
            result.append(item)
    print(result)
    bonus1 = 100000 * 0.1
    bonus2 = bonus1 + 100000 * 0.500075
    bonus4 = bonus2 + 200000 * 0.5
    bonus6 = bonus4 + 200000 * 0.3
    bonus10 = bonus6 + 400000 * 0.15

    i = int(input('input gain:\n'))
    if i <= 100000:
        bonus = i * 0.1
    elif i <= 200000:
        bonus = bonus1 + (i - 100000) * 0.075
    elif i <= 400000:
        bonus = bonus2 + (i - 200000) * 0.05
    elif i <= 600000:
        bonus = bonus4 + (i - 400000) * 0.03
    elif i <= 1000000:
        bonus = bonus6 + (i - 600000) * 0.015
    else:
        bonus = bonus10 + (i - 1000000) * 0.01
    print('bonus = ', bonus)
    year = int(input('year:\n'))
    month = int(input('month:\n'))
    day = int(input('day:\n'))

    months = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
    if 0 <= month <= 12:
        sum = months[month - 1]
    else:
        print('data error')
    sum += day
    leap = 0
    if (year % 400 == 0) or ((year % 4 == 0) and (year % 100 != 0)):
        leap = 1
    if (leap == 1) and (month > 2):
        sum += 1
    print('it is the %dth day.' % sum)
    h = 0
    leap = 1
    for m in range(101, 201):
        k = int(sqrt(m + 1))
        for i in range(2, k + 1):
            if m % i == 0:
                leap = 0
                break
        if leap == 1:
            h += 1
            if h % 10 == 0:
                print('')
        leap = 1


def optimal_portfolio(returns_sim,n_portfolio): 
    #求解优化组合
    n = returns_sim.shape[1]
    ret_mean = matrix(np.mean(returns_sim,axis = 0))
    ret_min = minrisk_portfolio(returns_sim)[0]
    ret_max = max(ret_mean)
    #mus预期收益率水平
    mus = np.linspace(ret_min,ret_max,num = n_portfolio,endpoint = False)
    #print('min return:',mus[0],'max return:',mus[-1])

    P = matrix(np.cov(returns_sim.T))
    q = matrix(0.0,(n,1))
    G = -matrix(np.eye(n))
    h = matrix(0.0, (n ,1))
    A0 = matrix(1.0,(1,n))
    b0 = matrix(1.0)

    weights = [solvers.qp(P, q, G, h, matrix([A0,ret_mean.T]), matrix([b0,matrix(mu)]))['x'] 
                  for mu in mus]
                      
    #rets = [blas.dot(ret_mean.T,x) for x in weights]   #和mus相同
    risks = [np.sqrt(blas.dot(x.T,P*x)) for x in weights]  #x:n*1 
    return pd.DataFrame(mus,columns=['mean']),pd.DataFrame(risks,columns=['std']),pd.DataFrame(weights)


def mentocarlo(ret_mean,ret_cov,n_sim):
    #生成模拟数据，n_sim条有效边界，计算平均权重
    rets_total    = pd.DataFrame()
    risks_total   = pd.DataFrame()
    weights_total = pd.DataFrame()
    for i in range(n_sim):
        returns_sim = portsim(ret_mean,ret_cov,252,1)
        rets,risks,weights = optimal_portfolio(returns_sim,60)
        if rets['mean'][0] < 0 : #剔除最小预期收益组合为负的有效边界
            pass
        else:
            weights['label'] = weights.index.values 
            weights_total = weights_total.append(weights,ignore_index = True)
            rets_total = rets_total.append(rets,ignore_index = False)
            risks_total = risks_total.append(risks,ignore_index = False)
        
    weights_avg = weights_total.groupby('label').mean()
    return rets_total,risks_total,weights_avg


def resampleportfolio(returns):
    ret_mean = np.mean(returns,axis = 0)
    ret_cov  = np.cov(returns.T)
    ret_corr = np.corrcoef(returns.T)
    print(ret_corr)
    rets_total,risks_total,weights_avg = mentocarlo(ret_mean,ret_cov,2)
    
    mu   = np.asmatrix(ret_mean)
    cov  = np.asmatrix(ret_cov)
    ws   = np.asmatrix(weights_avg)
 
    rets_resample  = ws*mu.T
    risks_resample = np.sqrt(ws*cov*ws.T)
    number = random.randint(1, 10)
    risks_resample = [risks_resample[i,i] for i in range(len(risks_resample))]
    plt.plot(risks_total.values, rets_total.values, 'o',markersize=2) #n_sim有效边界
    plt.xlabel('std') # 标准差-波动性
    plt.ylabel('mean') # 平均值-收益率
    plt.plot(risks_resample, rets_resample, 'y-o',markersize=2) #n_sim有效边界
    #plt.title('重抽样有效边界')
    weights_avg.plot(kind ='area',stacked = True,ylim =[0,1],title ='重抽样权重')  #DataFrame.plot
    return number


if __name__ == '__main__':
    resampleportfolio(returns)