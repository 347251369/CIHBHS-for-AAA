import numpy as np
from queue import *
import scipy.linalg as la
import qiskit.quantum_info as qi
import collections

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, Aer, BasicAer
from math import pi
from qiskit.tools.visualization import plot_histogram


def TridiagonalToeplitz(power, x, y):
	size = 2**power
	A = np.eye(size)
	A = x*A
	for i in np.arange(0,size-1,1):
		A[i,i+1] = y
		A[i+1,i] = y
	return A


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

	for i in np.arange(0,n,1):
		circ.x(i)
	circ.mcx(control,n)
	for i in np.arange(0,n,1):
		circ.x(i)
	circ.mcx(control,n)

	circ.x(n)
	return circ

def Hamiltonian1():
	q = QuantumRegister(2)
	circ = QuantumCircuit(q)

	A = TridiagonalToeplitz(2, 1, 1)
	e_U = la.expm(1j*A*t)
	circ.unitary(e_U,q)
	U_gate = circ.to_gate().control(1)

	p = QuantumRegister(3)
	cir = QuantumCircuit(p)
	cir.h(0)
	cir.append(U_gate,[0]+p[1:3])

	return cir

def Hamiltonian2(m,t):
	q = QuantumRegister(3)
	circ = QuantumCircuit(q)

	
	cir2 = A2(2,1/m,t).to_gate()
	cir3 = A3(2,1/m,t).to_gate()
	ctl = list(range(0, 2))
	for i in np.arange(0,m,1):
		circ.append(cir2,ctl)
		circ.append(cir3,ctl+[2])
	
	U_gate = circ.to_gate().control(1)

	p = QuantumRegister(1+2+1)
	cir = QuantumCircuit(p)
	cir.h(0)
	cir.p(t,0)
	cir.append(U_gate,[0]+p[1:1+2+1])

	return cir

def complex_vector_similarity(v1, v2):
    product = np.dot(v1.conj(), v2)
    length_v1 = np.linalg.norm(v1)
    length_v2 = np.linalg.norm(v2)
    similarity = product / (length_v1 * length_v2)
    return similarity
 

t = pi/1
power = 2
n = 4

A = TridiagonalToeplitz(2, 1, 1)
u1 = Hamiltonian1()
q1 = QuantumRegister(3)
cir1 = QuantumCircuit(q1)
cir1.append(u1,q1)
simulator1 = BasicAer.get_backend('statevector_simulator')
job1 = execute(cir1, simulator1)
result1 = job1.result()
wavefunction1 = result1.get_statevector(cir1)


result = []
for m in np.arange(1,11,1):
	u2 = Hamiltonian2(m,t)
	q2 = QuantumRegister(4)
	cir2 = QuantumCircuit(q2)
	cir2.append(u2,q2)
	simulator2 = BasicAer.get_backend('statevector_simulator')
	job2 = execute(cir2, simulator2)
	result2 = job2.result()
	wavefunction2 = result2.get_statevector(cir2)

	f_ave = complex_vector_similarity(wavefunction1[0:8], wavefunction2[0:8])
	ans = np.linalg.norm(f_ave)
	print("Average Fidelity: F = {:f}".format(ans))
	result.append(ans)
	
np.save("data"+str(t)+".npy", result)