#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
n = 3
m = 2
np.random.seed(1)
amat = np.random.uniform(size = (n,m))
amat[0,1]
amat[0,:]
amat[:,1]

import numpy as np
n = 3
m = 2
np.random.seed(1)
amat = np.random.uniform(size = (n,m))
amat
bvec = np.array([0.1,0.5]).reshape(m,1)
bvec
amat@bvec

# columnwise operation
y1 = amat@bvec
y1.squeeze()
amat[:,0]*bvec[0]+amat[:,1]*bvec[1]


# 행렬의 덧셈과 선형변환
np.random.seed(1)
amat_1 = np.random.uniform(size = (n,m))
amat_2 = np.random.uniform(size = (n,m))
amat_3 = amat_1 + amat_2
bvec = np.random.uniform(size = (m,1))
print( amat_1@bvec + amat_2@bvec,'\n' )
print (amat_3@bvec)

# columnwise operation
n = 10
p = 5
np.random.seed(1)
x = np.random.normal(size = (n,p))
betavec = np.random.uniform(size = (p,1))
y = x@betavec
z = np.zeros(n)
for i in range(p):
    z += x[:,i]*betavec[i] 
z = z.reshape(-1,1)
np.concatenate((y,z), axis = 1)

# 샘플 데이터를 불러보자
import pandas as pd
path = "C:/Users/jjjeo/Desktop/1인가구_시각화_교육용2/"
data = pd.read_csv(path+'sample.csv', encoding='cp949')
data.head()
n = data.shape

# 코드에 대한 설명 파일을 불러보자
code_data = pd.read_csv(path + 'code_pre.csv', 
                        encoding = 'cp949')
# 코드에 대한 설명을 확인하자
code_data['속성명']

# 분석 대상을 정의하자
features = [code_data['속성명'][i] for i in [4,5,6,7,9]]
features
# 필요한 데이터 컬럼의 추출
rdata = data.iloc[:,[4,5,6,7,9]]
x = np.array(rdata)
n, p = x.shape
# 데이터정규화
from sklearn.preprocessing import StandardScaler 
x = StandardScaler().fit_transform(x) 
x.mean(axis=0)
x.std(axis=0)

bvec = np.random.uniform(size = p)
bvec=bvec/np.linalg.norm(bvec)
(x[0,:]*bvec).sum()
y = x@bvec
plt.hist(y, bins = 200, range = (-10,10))

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(x)
# 설명력
pca.explained_variance_ratio_
# 
pca.singular_values_

pc_score = pca.fit_transform(x)
pc_score
import matplotlib.pyplot as plt
plt.plot(pc_score[:,0], pc_score[:,1], '.') 
x.shape

# care about full_matrices = False
u, s, vh= np.linalg.svd(x, full_matrices=False)
u.shape
s.shape
vh.shape
(vh[0,:]*vh[1,:]).sum()


# UDV^T 에서 V^T matrix를 확인
# V행렬의 열벡터는 직교함

# 해석
features 
vh[0,:]
vh[1,:]
# 0번 obs의 표현형
u[0,:]
# scaled
plt.plot(u[:,0],u[:,1], '.')

# considering the size s
plt.plot(s[0]*u[:,0],s[1]*u[:,1], '.')


# approximation
k = 1
x_approx = u[:,:k]@np.diag(s[:k])@vh[:k,:]
print( np.mean((x-x_approx)**2) )

# why covariance
u@np.diag(s)@vh
# x^T
vh.T@(np.diag(s)).T@u.T

vh.T@(np.diag(s)).T@u.T@u@(np.diag(s))@vh
#
x.T@x
# covariance
x.T@x/x.shape[0]

covmat = np.cov(x, rowvar = False, bias = True)
covmat
((x.T)@x)/(x.shape[0])

# covariance decomposition with eigendecomposition
# eigenvector
dvec, emat = np.linalg.eig(covmat)
# eigenvalue
dvec
# eigenvector
emat[:,0]
s**2/x.shape[0]

# 실습 app 사용에 대한 지수를 만들어 보자
code_data.columns
features


# for 문 연습하기
list_a = ['Buddy,', 'you\'re', 'a boy', 'make', 'a big noise']

enumerate(list_a)

zip(range(len(list_a)), list_a)

for j,k in enumerate(list_a):
    print(j, k)

for j,k in zip(range(len(list_a)), list_a):
    print(j, k)

# step by step

it = enumerate(list_a)
next(it)

it = zip(range(len(list_a)), list_a)
next(it)

it = iter(list_a)
next(it)