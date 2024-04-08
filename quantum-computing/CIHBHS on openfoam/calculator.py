import numpy as np
from queue import *
import scipy.linalg as la
import qiskit.quantum_info as qi
import collections
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, Aer

def TridiagonalToeplitz(power, x, y):
	size = 2**power
	A = np.eye(size)
	A = x*A
	for i in np.arange(0,size-1,1):
		A[i,i+1] = y
		A[i+1,i] = y
	return A

def Normalize(a):
	coef = a.dot(a)**(0.5)
	return [coef, a/coef]

def get_dig(a):
	dig = 0
	while a != 0:
		a = int(a/2)
		dig = dig + 1
	return dig 

def get_pos_from_high_to_low(a):
	binary = bin(a)[2:]
	if len(binary)==1:
		return []
	else:
		s = binary[1:]
		pos = []
		for i, c in enumerate(s):
			if c == '0':
				pos.append(i)
		return pos


def get_control(i,n):
	dig = get_dig(i)
	dig = dig -1
	control = []
	for j in np.arange(0,dig,1):
		control.append(n-1-j)
	return control



def Quantum_states(b,n):
	circ = QuantumCircuit(QuantumRegister(n))
	length = b.shape[0]
	q = Queue()
	q.put(b*b)
	for i in np.arange(1,length,1):
		temp_b = q.get()
		len_b = temp_b.shape[0]
		half = int(len_b/2)
		sum_half1 = sum(temp_b[:half])**(0.5)
		sum_half2 = sum(temp_b[half:])**(0.5)
		theta = np.arctan2(sum_half2, sum_half1)

		dig = get_dig(i)
		pos = get_pos_from_high_to_low(i)
		if dig == 1:
			circ.ry(2*theta,n-1)
		else:
			for j in pos:
				circ.x(n-1-j)
			control = get_control(i,n)
			circ.mcry(2*theta,control,n-dig)
			for j in pos:
				circ.x(n-1-j)
		if half > 1:
			q.put(temp_b[:half])
			q.put(temp_b[half:])
	circ.barrier()
	return circ



def A1(n,t):
	q = QuantumRegister(n)
	circ = QuantumCircuit(q,name='A1')

	circ.p(t,0)
	circ.x(0)
	circ.p(t,0)
	circ.x(0)
	return circ

def A2(n,y,t):
	q = QuantumRegister(n)
	circ = QuantumCircuit(q,name='A2')

	circ.rx(-2*y*t,0)
	return circ

def add1(n):
	q = QuantumRegister(n)
	circ = QuantumCircuit(q,name='add')

	control = list(range(0, n-1))

	for i in np.arange(n-1,0,-1):
		circ.mcx(control,i)
		control.pop(i-1)
	circ.x(0)

	return circ

def A3(n,y,t):
	q = QuantumRegister(n+1)
	circ = QuantumCircuit(q,name='A3')

	circ.x(n)
	control = list(range(0, n))
	circ.mcx(control,n)
	for i in np.arange(0,n,1):
		circ.x(i)
	circ.mcx(control,n)
	for i in np.arange(0,n,1):
		circ.x(i)

	add_cir = add1(n)
	add_gate = add_cir.to_gate().control(1)

	circ.append(add_gate,[n]+control)

	circ.crx(-2*y*t,n,0)
	minu_gate = add_cir.inverse().to_gate().control(1)
	circ.append(minu_gate,[n]+control)

	circ.mcx(control,n)
	for i in np.arange(0,n,1):
		circ.x(i)
	circ.mcx(control,n)
	for i in np.arange(0,n,1):
		circ.x(i)

	return circ

def Hamiltonian(n,t,y):
	q = QuantumRegister(n)
	circ = QuantumCircuit(q)

	A = TridiagonalToeplitz(n,1,y)
	e_U = la.expm(1j*A*t)
	circ.unitary(e_U,q)

	return circ

def qft(circ,c,cn):
	for i in np.arange(0,cn//2,1):
		circ.swap(c[i],c[cn-1-i])
	for i in np.arange(0,cn,1):
		for j in np.arange(0,i,1):
			circ.cp(-np.pi/float(2**(i-j)),c[j],c[i])
		circ.h(c[i])


def Quantum_phase_estimate(cn,bn,t,y):
	q = QuantumRegister(cn+bn)
	circ = QuantumCircuit(q)
	c = q[:cn]
	b = q[cn:cn+bn]
	d = q[cn+bn:cn+bn+1]

	for i in np.arange(0,cn,1):
		circ.h(c[i])

	U_gate = Hamiltonian(bn,t,y).to_gate().control(1)

	repetitions = 1
	for c_i in range(0,cn,1):
		for j in range(repetitions):
			circ.append(U_gate,[c_i]+b)
		repetitions = repetitions * 2

	circ.barrier()
	qft(circ,c,cn)
	circ.barrier()
	return circ

def get_pos_from_low_to_high(a):
	dig = get_dig(a)
	binary = bin(a)[2:]
	if len(binary)==1:
		return []
	else:
		s = binary[1:][::-1]
		pos = []
		for i, c in enumerate(s):
			if c == '0':
				pos.append(i)
		return pos

def Control_rotate(cn):
	q = QuantumRegister(1+cn)
	circ = QuantumCircuit(q)
	a = q[:1]
	c = q[1:]

	max_n = 2**cn
	half = int(max_n/2)
	for i in np.arange(1,max_n,1):
		dig = get_dig(i)
		pos = get_pos_from_low_to_high(i)
		for j in np.arange(dig,cn,1):
			pos.append(j)
		for j in pos:
			circ.x(c[j])
		if i < half:
			theta = 2*np.arcsin(1/i)
		else:
			theta = 2*np.arcsin(1/(i-max_n))
		circ.mcry(theta,c,a[0])
		for j in pos:
			circ.x(c[j])

	return circ


def HHL(power,y,B,cn,shots):
	A = TridiagonalToeplitz(power, 1, y)
	[norm_coef, B] = Normalize(B)
	eigenvalues = abs(la.eigvals(A).real)
	eigenvalues_sort = np.sort(eigenvalues)


	[an,bn] = [1, int(np.log2(B.shape[0]))]
	q = QuantumRegister(an+cn+bn)
	circ = QuantumCircuit(q)
	cr = ClassicalRegister(an+bn,'creg')
	circ.add_register(cr)
	a = q[0:an]
	c = q[an:an+cn]
	b = q[an+cn:an+cn+bn]
	t = 1/(2**cn)*2*np.pi/eigenvalues_sort[0]


	state_b = Quantum_states(B, bn)
	circ.append(state_b, b)

	qpe = Quantum_phase_estimate(cn,bn,t,y)
	circ.append(qpe,c+b)


	ctlr = Control_rotate(cn)
	circ.append(ctlr,a+c)

	inv_qpe = qpe.inverse()
	circ.append(inv_qpe,c+b)

	for i in np.arange(bn-1,-1,-1):
		circ.measure(b[bn-1-i],cr[bn-1-i])
	circ.measure(a[0],cr[bn])

	simulator = Aer.get_backend('aer_simulator')
	job = execute(circ, simulator, shots=shots, memory=True)
	result = job.result()
	memory = result.get_memory(circ)

	for i in np.arange(len(memory)-1,-1,-1):
		if memory[i][0]== '0':
			del memory[i]
		else:
			memory[i]=memory[i][1:]

	results = collections.Counter(memory)
	results = sorted(results.items(), key=lambda x: x[0], reverse=False)
	
	x = []
	num = 0
	for item in results:
		key = item[0]
		value = item[1]
		while num < int(key, 2):
			x.append(0)
			num = num + 1
		x.append(value)
		num = num + 1
	while num<len(B):
		x.append(0)
		num = num + 1

	x = np.array(x)
	[norm_coef_, x] = Normalize(x**0.5)

	return x


def CIHBHS(power, scale, y, b, cn, shots, times, learning_rate):
	A = TridiagonalToeplitz(power, 1, y)
	x_true = np.linalg.solve(A,b)
	x = 0
	b_j = b
	x_error = []
	for j in np.arange(1,times+1,1):
		[norm_coef, b_j] = Normalize(b_j)
		x_hhl = HHL(power,y,b_j,cn,shots)
		b_ = np.dot(A, x_hhl)

		x_r = 1/np.linalg.norm(b_)*x_hhl*norm_coef
		error = x + x_r - x_true
		error = (np.dot(error,error)/error.shape[0])**0.5
		print(error)
		x_error.append(error)

		if j == times:
			x = x + x_r
		else:
			rate = float('inf')
			for i in np.arange(0,b.shape[0],1):
				rate = min(rate,b_j[i]/b_[i])
			rate = rate *np.linalg.norm(b_) * learning_rate
			x = x + x_r * rate

		b_j = b - np.dot(A, x)

	x_error = np.log10(np.array(x_error))
	return [x_true, x*scale, x_error]


def construct_AB(A, b):
	A = A.split(' ')
	b = b.split(' ')
	
	b = np.array(list(map(float,np.array(b))))
	n = b.shape[0]
	A = np.array(list(map(float,np.array(A))))
	return n, A, b

def cal(A, b, p):
	n, A, b = construct_AB(A, b)
	print(n)
	print(A)
	print(b)
	power = int(np.log2(n))
	scale = A[0]
	y = A[1]/A[0]
	cn = 5
	shots = 10000
	times = 10
	learning_rate = 0.5

	[x_true, x, x_error] = CIHBHS(power, scale, y, b, cn, shots, times, learning_rate)
	s = ""
	for i in range(0, len(x)):
		s = s + str(x[i]) + ' '
	return s