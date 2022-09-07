import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

os.getcwd()
fname_list = os.listdir("./data")
fname = fname_list[0]
tmp = pd.read_csv("./data/"+ fname, 
                    index_col = False, 
                    encoding='utf-8')
rdata = tmp.iloc[:,0:4]
dong_code = rdata["행정동코드"].unique()
len(dong_code) # 행정동 424개

# pd.pivot 을 이용하여 행정동을 열로 표시하고 총생활인구를 값으로
rdata_w = pd.pivot(rdata, index = ["기준일ID", "시간대구분"],
                   columns="행정동코드", values='총생활인구수')
rdata_w.head()
rdata_w.reset_index(inplace=True)
rdata_w.columns
n, p = rdata_w.shape

# 시간에 대한 계산 (특정일 이전 시간대를 검색하여 지우기 등..)
# 현재 시간정보의 확인
rdata_w['기준일ID']
rdata_w['시간대구분']
# 두 시간을 string으로 합친후에 시간으로 변경하자
# str formating
'정수 3자리:{0:03d},{1:03d}'.format(123,12)
'{0:02d}'.format(0)
tmp1 = [ str(i) for i in list(rdata_w['기준일ID'])]  
tmp2 = ['{0:02d}'.format(i) for i in list(rdata_w["시간대구분"])]
tmp3 = [i + j for i,j in zip(tmp1, tmp2)]
tmp3 = pd.to_datetime(tmp3, format ="%Y%m%d%H")
rdata_w["date"] = tmp3
rdata_w.shape


#codes, uniques = rdata_w['기준일ID'].factorize()
# slicing time
v = pd.to_datetime("2022070400", format ="%Y%m%d%H")
idx = (tmp3 > v)
n = sum(idx)
idx_c = np.repeat(True,rdata_w.shape[1])
idx_c[0:2] = np.array([False, False])
y = rdata_w.loc[idx,idx_c]
x = rdata_w.iloc[0:y.shape[0],:]
y.shape
x.shape


# 총인구 확인해보기
# 일별 유동인구의 수 변화 
total_pop_y = y.sum(axis = 1)
plt.plot(total_pop_y)
plt.figure(figsize=(30,10))
plt.plot(total_pop_y)

# 행정동별  최대인구
total_pop_ys = y.mean(axis = 0)
plt.figure(figsize=(30,10))
plt.plot(total_pop_ys)
plt.bar(range(len(total_pop_ys)), total_pop_ys)
total_pop_ys.idxmax()

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
district_code_info.head()

max_idx = total_pop_ys.idxmax()
idx = (district_code_info['H_DNG_CD'] == max_idx)
if sum(idx)>=1:
    print(district_code_info['district'][idx])
else:
    print('찾은 행정동 코드 '+ str(max_idx) +  '가 DB에 없습니다. ')
    

# 숫자 객체를 다룰때 numpy로 변환후에 사용
np.eye(3,2)
np.identity(3)
np.empty((3,3))


# 4일전 인구정보 저장하기
x.columns[2:426]
len(x.columns[2:426].unique())
pop = np.array(x[x.columns[2:426]])
pop.shape
pop_index = list( x.columns[2:426] )
# 날짜를 숫자벡터로 변경하기
# weekday index
# mon to sun: 0 to 6
weekday = [ x['date'][i].dayofweek for i in range(n)]

weekday_mat = np.zeros(shape = (n,7))
for i, j in zip(range(n), weekday):
    weekday_mat[i,j] = 1
weekday_mat.shape
weekday_index = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
# 시간을 숫자벡터로 변경하기
hour = x['시간대구분'] 
hour_mat = np.zeros(shape = (n,24))
for i, j in zip(range(n),hour):
    hour_mat[i,j] = 1
hour_mat.shape
hour_index = [str(i) + 'h' for i in range(24)]
# predictor 데이터 합치기 (4일전 인구, 요일, 시간)
x_index = pop_index + weekday_index + hour_index
x_ = np.concatenate([pop, weekday_mat, hour_mat], axis = 1)
x_.shape

# response 데이터 합치기
y_ = y.drop("date", axis = 1)
y_ = np.array(y_)
pop_y_index = ["y"+ str(i)  for i in pop_index]

# 데이터 저장하기
rdata_y = pd.DataFrame(y_, columns = pop_y_index)
rdata_x = pd.DataFrame(x_, columns = x_index)

rdata_y.to_csv("./real/y.csv", index=False)
rdata_x.to_csv("./real/x.csv", index=False)