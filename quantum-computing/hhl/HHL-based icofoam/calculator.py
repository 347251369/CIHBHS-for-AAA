import numpy as np
import pyqpanda as pq

def add(a, b):
    print(a + b)
    return a + b

def helloworld(s):
    print("hello " + s)

def pri(sa, sb):
  A = sa.split(' ')
  B = sb.split(' ')
  c = np.array(A)
  s = "Python-check"
  for i in B:
      s = s + str(i) + ' '
  return s

def construct_AB(A, b):
	A = A.split(' ')
	b = b.split(' ')
	n = len(b)
	A = np.array(A).reshape(n,n).tolist()
	length = 1
	while length < n:
		length *= 2
	if length > n:
		for i in range(n):
			for j in range(n, length):
				A[i].insert(0, 0)
	for i in range(n, length):
		A.append([0 for j in range(length)])

	HA = [[0 for j in range(length*2)] for i in range(length*2)]
	for i in range(length):
		for j in range(length):
			HA[i][j+length] = A[i][j]
			HA[i+length][j] = A[j][i]
	for i in range(n, length):
		HA[i][length*2-1-i] = HA[length*2-1-i][i] = 1
	HA = [b for a in HA for b in a]

	Hb = [x for x in b]
	for i in range(n, length*2):
		Hb.append(0)

	return n, list(map(float,HA)), list(map(float,Hb))

def cal(A, b, p):
	n, HA, Hb = construct_AB(A, b)
	result = pq.HHL_solve_linear_equations(HA,Hb,p)
	s = ""
	for i in range(len(Hb)-n, len(Hb)):
		s = s + str(result[i].real) + ' '
	return s