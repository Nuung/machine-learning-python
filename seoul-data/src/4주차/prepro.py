
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


os.getcwd()
fname_list = os.listdir("./data") #파일이름저장
file_sel_index = [3,4,5,6]
i = 0 ; j = 3
for i,j in enumerate(file_sel_index):
    fname = fname_list[j]
    tmp = pd.read_csv("./data/"+ fname, 
                        index_col = False, 
                        encoding='cp949')
    idx1 = (tmp['요일'] == '금') | (tmp['요일'] == '토')
    idx2 = (tmp['나이'] >= 20) & (tmp['나이'] <= 30)
    idx = idx1 & idx2
    tmp1 = tmp.loc[idx,:]
    # 출발 행정동 코드, 도착 행정동 코드, 이동인구(합)
    tmp1 = tmp1.iloc[:,[3,4,9]]
    idx = tmp1['이동인구(합)']=="*"
    tmp1['이동인구(합)'][idx] = 1.0
    tmp1['이동인구(합)'] = tmp1['이동인구(합)'].astype('float32')
    idx = tmp1['이동인구(합)']>=1
    tmp1 = tmp1.loc[idx,:]
    tmp1 = tmp1.groupby(['출발 행정동 코드', '도착 행정동 코드'])['이동인구(합)'].sum()
    tmp1 = tmp1.reset_index()
    if i==0:
        rdata = tmp1
        next
    rdata = pd.concat([rdata, tmp1], axis = 0)

rdata.head()
rdata_L = rdata.groupby(['출발 행정동 코드', '도착 행정동 코드'])['이동인구(합)'].sum()
rdata_L = rdata_L.reset_index()
rdata_L.head()
rdata_L.shape


dong_code1 = set(rdata_L["출발 행정동 코드"].unique())
dong_code2 = set(rdata_L["도착 행정동 코드"].unique())
dong_code = list( set.union(dong_code1, dong_code2) )
dong_code.sort()
p = len(dong_code)
dict_dong_index = dict(zip(dong_code, range(p)))
dict_index_dong = dict(zip(range(p), dong_code))

n = rdata_L.shape[0]
start_loc = np.array(rdata_L['출발 행정동 코드'], dtype ='int64')
end_loc = np.array( rdata_L['도착 행정동 코드'], dtype ='int64')
pop = np.array(rdata_L['이동인구(합)'], dtype='float32')
trans_mat = np.zeros(shape = (p,p))
i = 0
for i in range(n):
    k1 = dict_dong_index.get(start_loc[i])
    k2 = dict_dong_index.get(end_loc[i])
    trans_mat[k1,k2] = pop[i] 
    if i%1000==0:
        print(i)

plt.plot(trans_mat.sum(axis = 1))
rowsums = trans_mat.sum(axis = 1)
sum(rowsums==0)
trans_prob = (trans_mat.T/rowsums).T
trans_prob[rowsums==0, :] = 1/p
plt.plot(trans_prob[0,:])

a = 0.95
modif_prob = np.ones((p,p))/p
trans_prob_m = a*trans_prob + (1-a)* modif_prob
# check: 행의 합의 0인 것이 있는지?
np.isnan(trans_prob_m).sum()

iter=10000
np.random.seed(1)
init_prob =np.random.dirichlet(np.ones(p),1)
init_prob.shape
#trans_prob.sum(axis = 1).min()
for i in range(iter):
    if i==0:
        lim_prob = init_prob@trans_prob_m    
    else:
        old_prob = lim_prob
        lim_prob = lim_prob@trans_prob_m
    if (i%10==0):
        err = np.max(abs(lim_prob-old_prob))
        print('error:', err)
        prob_sum = lim_prob.sum()
        print('prob:', prob_sum)

sum(lim_prob)
lim_prob = lim_prob.reshape(-1)
plt.plot(lim_prob)

max(lim_prob)
1/p
np.sort(lim_prob)[-1:-11:-1]
idx = np.argsort(lim_prob)[-1:-11:-1]
cent_idx = [ dict_index_dong[i] for i in idx ]      


# 행정동 코드 확인하기
col_index = pd.read_csv("./dong_code.csv", nrows = 2, 
                        header = None, encoding='cp949')
district_code_info = pd.read_csv("./dong_code.csv", skiprows=1,
                 encoding='cp949')
tmp = []
for i in range(district_code_info.shape[0]):
    v = district_code_info['CT_NM'][i] + " " + district_code_info['H_DNG_NM'][i] 
    tmp.append(v)
tmp
district_code_info['district'] = tmp

rank = 4
idx = (district_code_info['H_SDNG_CD'] == cent_idx[rank])
if sum(idx)>=1:
    print(district_code_info['district'][idx])
else:
    print('찾은 행정동 코드 '+ str(max_idx) +  '가 DB에 없습니다. ')

