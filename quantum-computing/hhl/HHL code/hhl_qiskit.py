from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, Aer
from math import pi
from qiskit.tools.visualization import plot_histogram
import warnings
warnings.filterwarnings('ignore')

q = QuantumRegister(4)
circ = QuantumCircuit(q)

circ.h(q[1])
circ.h(q[2])
circ.h(q[3])


circ.rz(pi/2, q[2])
circ.rz(pi, q[1])
circ.cx(q[2], q[3])


circ.cx(q[1], q[2])
circ.cx(q[2], q[1])
circ.cx(q[1], q[2])



circ.h(q[2])
circ.cx(q[1], q[2])
circ.rz(pi/4, q[2])
circ.cx(q[1], q[2])
circ.rz(-pi/4, q[2])
circ.rz(-pi/4, q[1])
circ.h(q[1])


circ.x(q[1])

circ.ry(pi/16, q[0])
circ.cx(q[1], q[0])
circ.ry(-pi/16, q[0])
circ.cx(q[1], q[0])


circ.ry(pi/32, q[0])
circ.cx(q[2], q[0])
circ.ry(-pi/32, q[0])
circ.cx(q[2], q[0])


circ.x(q[1])
circ.h(q[1])


circ.cx(q[1], q[2])
circ.rz(-pi/4, q[2])
circ.cx(q[1], q[2])
circ.rz(pi/4, q[2])
circ.rz(pi/4, q[1])


circ.h(q[2])
circ.cx(q[1], q[2])
circ.cx(q[2], q[1])
circ.cx(q[1], q[2])

circ.cx(q[2], q[3]) 
circ.rz(-pi, q[1])
circ.rz(-pi/2, q[2])

circ.measure_all()


simulator = Aer.get_backend('aer_simulator')
job = execute(circ, simulator, shots=8000, memory=True)
result = job.result()

memory = result.get_memory(circ)

x = {'0':0,'1':0}
for i in memory:
	if i[3] == '1':
		if i[0] == '0':
			x['0'] = x['0'] + 1
		else:
			x['1'] = x['1'] + 1

p = x['0']/x['1']
print('number of 0:', x['0'])
print('number of 1:', x['1'])
print('the ratio of two probabilities', p)


plot_histogram(x, title='Bell-State counts').savefig('out0.png')