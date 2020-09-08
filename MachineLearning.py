import numpy as np

A=[[2, 3], [4, 5]]
B=[[2, 3], [4, 5]]
ANP=np.array(A)
BNP=np.array(B)
print(ANP)
ANP_transpose=ANP.T
print(ANP_transpose)

# 1) np.dot(x, y) : 행렬의 내적
print("1) np.dot(x, y) : 행렬의 내적")
C=np.dot(ANP, BNP)
print(C)
D1=np.dot(ANP_transpose, BNP)
print(D1)
D2=ANP.T.dot(BNP)
print(D2)
print("\n")

# 2) np.diag : 대각행렬
print("2) np.diag : 대각행렬")
DiagonalA=np.diag(A)
DiagonalB=np.diag(B)
print(DiagonalA)
print(DiagonalB)
print(np.diag(ANP_transpose))
print(np.diag(C))
print(np.diag(D1))
print(np.diag(D2))
print("\n")

# 3) np.trace : 대각합
print("3) np.trace : 대각합")
traceA=np.trace(A)
traceB=np.trace(B)
print(traceA)
print(traceB)
print(np.trace(C))
print(np.trace(D1))
print(np.trace(D2))
print("\n")

# 4) np.linalg.det : 행렬식
print("4) np.linalg.det : 행렬식")
detA=np.linalg.det(A)
detB=np.linalg.det(B)
print(detA)
print(detB)
print(np.linalg.det(C))
print(np.linalg.det(D1))
print(np.linalg.det(D2))
print("\n")

# 5) np.linalg.inv : 역행렬
print("5) np.linalg.inv : 역행렬")
invA=np.linalg.inv(A)
invB=np.linalg.inv(B)
print(invA)
print(invB)
print(np.linalg.inv(C))
print(np.linalg.inv(D1))
print(np.linalg.inv(D2))
print("\n")

# 6) np.linalg.svd : 특이값 분해
print("6) np.linalg.svd : 특이값 분해")
b=[[0, 0], [0, 0]]
E=A+b
arrayE=np.array(E)
print(arrayE)
u, s, vh=np.linalg.svd(arrayE)
print(u)
print(s)
print(vh)
print("\n")

# 7) np.linalg.solve : 연립방정식 해 풀기
print("7) np.linalg.solve : 엽립방정식 해 풀기")
F=np.array([5, 7])
X=np.linalg.solve(ANP, F)
print(X)

va=np.array([1,2,3])
vb=np.array([4,5,6])
print(va)
xr=np.r_[va, vb]
xc=np.c_[va,vb]
print(xr)
print(xc)