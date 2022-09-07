# 필요 파일
# - 1인가구.csv, data.shp, 
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# OS가 window인 경우 geopanda 설치 오류가 발생하는 경우
# 필수 의존성 패키지 설치를 확인 
# 1. pyproj, 2. Shapely,  3. GDALL, 4.Fiona

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc('font', family='Malgun Gothic') # 한글 폰트 적용
plt.rcParams["figure.figsize"] = (20, 10) # 차트 사이즈
mpl.rcParams["axes.unicode_minus"] = False

"""1인가구 지수 데이터"""
# 데이터 형식의 통일
path = "C:/Users/jjjeo/Desktop/1인가구_시각화_교육용2/1인가구.csv"
data = pd.read_csv(path, encoding='cp949')
data.head()
n = data.shape[0]
data.columns
adm_cd = []
for i in range(n):
    adm_cd.append(str(data['행정동코드'][i]))
data['adm_cd'] = adm_cd

"""행정동 위치 파일과 결합"""
geo = gpd.read_file('data.shp')
geo.head()
df_geo = geo.iloc[:,[2,9]]
rdata = pd.merge(data, df_geo, on = 'adm_cd')
rdata.head()
type(rdata)
data_merge = gpd.GeoDataFrame(rdata, crs="EPSG:4326", geometry="geometry")
type(data_merge)
"""시각화: 20~24세, 남성 평일 외출이 적은 집단"""
# 추출
data_tmp = data_merge[data_merge['연령대'] == 20][data_merge['성별'] == 1]
idx = (data_merge['연령대'] == 20) & (data_merge['성별'] == 1)
data_tmp.loc[idx,:]
data_tmp.head()
type(data_tmp)
# 시각화,  column 에 시각화할 필드명을 입력
fig = plt.figure()
data_tmp.plot(column='평일 외출이 적은 집단',
                 legend=True,
                 cmap='YlGn',
                 edgecolor='k',
                 legend_kwds={'label': '명'})
plt.axis('off')
plt.tight_layout()
plt.savefig('example_1.png')
plt.show()

# 연령대의 병합
# 20-29: 20대 (초기청년층)
# 30-39: 30대 (후기청년층)
# 40-49: 40대 (중년층)
# 50-64: 50대 (장년층)
# 65: 65세 이상 (노년층)
data_merge['연령대1'] = data_merge['연령대']

idx = (data_merge["연령대"] == 25)
data_merge["연령대1"][idx] = 20

idx = (data_merge["연령대"] == 35)
data_merge["연령대1"][idx] = 30

idx = (data_merge["연령대"] == 45)
data_merge["연령대1"][idx] = 40

idx = (data_merge["연령대"] == 55)
data_merge["연령대1"][idx] = 50

idx = (data_merge["연령대"] == 60)
data_merge["연령대1"][idx] = 50

idx = (data_merge["연령대"] > 60)
data_merge["연령대1"][idx] = 65

colname_ = list ( data_merge.columns[5:17] )
data_groupby = data_merge.groupby(["adm_cd",'연령대1'])[colname_].sum()
data_groupby.head()
data_groupby = data_groupby.reset_index()
data_groupby = pd.merge(data_groupby, df_geo, on = 'adm_cd')
data_groupby = gpd.GeoDataFrame(data_groupby, crs="EPSG:4326", 
                                geometry="geometry")

# 중년층 중 휴일 외출이 적은 집단
data_tmp = data_groupby[data_groupby['연령대1']==40]
fig = plt.figure()
data_tmp.plot(column='휴일 외출이 적은 집단',
                 legend=True,
                 cmap='YlGn',
                 edgecolor='k',
                 legend_kwds={'label': '명'})
plt.axis('off')
plt.tight_layout()
plt.savefig('example_2.png')
plt.show()



