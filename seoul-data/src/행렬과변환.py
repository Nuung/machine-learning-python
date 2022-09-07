# python lib
import numpy as np

n = 3
m = 2

# ex1
np.random.seed(1)
amat = np.random.uniform(size = (n, m))
print(amat)

bvec = np.array([0.1, 0.5]).reshape(m, 1)
print(bvec)
print(amat@bvec)

# ex 2
# 백터들의 원소들이 곱하기 접근하기
y1 = amat@bvec
y1.squeeze()
temp = amat[:,0]*bvec[0] + amat[:,1]*bvec[1]
print(temp)


# ex 3
# 행렬의 덧셈과 새로운 선형변환
np.random.seed(1)
amat_1 = np.random.uniform(size = (n, m))
amat_2 = np.random.uniform(size = (n, m))
amat_3 = amat_1 + amat_2
bvec = np.random.uniform(size = (m, 1))
print(amat_1@bvec + amat_2@bvec)
print(amat_3@bvec)

np.random.seed(1)
k = 5
A = np.random.uniform(size = (n, k))
print(A.shape)
B = np.random.uniform(size = (k, m))
print(B.shape)
print(A@B)
