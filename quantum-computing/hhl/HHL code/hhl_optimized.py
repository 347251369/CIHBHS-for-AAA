import numpy as np
from math import pi
#from Challenge_Account import *
from ezQpy import *

class Qgate():
    def __init__(self):
        self.fixed_X_gate = '''X Q%s
        '''
        self.fixed_H_gate = '''H Q%s
        '''
        self.fixed_X2M_gate = '''X2M Q%s
        '''
        self.fixed_X2P_gate = '''X2P Q%s
        '''
        self.fixed_Y2M_gate = '''Y2M Q%s
        '''
        self.fixed_Y2P_gate = '''Y2P Q%s
        '''
        self.fixed_RX_gate = '''RX Q%s %s
        '''
        self.fixed_RY_gate = '''RY Q%s %s
        '''
        self.fixed_RZ_gate = '''RZ Q%s %s
        '''
        self.fixed_CZ_gate = '''CZ Q%s Q%s
        '''
        self.fixed_measure_gate = '''M Q%s
        '''

    def ReArea(self, theta):
        theta = theta % (2*np.pi)
        if theta > np.pi:
            theta = theta - 2*np.pi
        return theta

    def X_gate(self, bit):
        return self.fixed_X_gate % (bit)

    def H_gate(self, bit):
        return self.fixed_H_gate % (bit)

    def X2M_gate(self, bit):
        return self.fixed_X2M_gate % (bit)

    def X2P_gate(self, bit):
        return self.fixed_X2P_gate % (bit)

    def SX_gate(self, bit):
        return self.fixed_X2P_gate % (bit)

    def Y2M_gate(self, bit):
        return self.fixed_Y2M_gate % (bit)

    def Y2P_gate(self, bit):
        return self.fixed_Y2P_gate % (bit)

    def SY_gate(self, bit):
        return self.fixed_Y2P_gate % (bit)

    def RX_gate(self, theta, bit):
        theta = self.ReArea(theta)
        return self.fixed_RX_gate % (bit, theta)

    def RY_gate(self, theta, bit):
        theta = self.ReArea(theta)
        return self.fixed_RY_gate % (bit, theta)

    def RZ_gate(self, theta, bit):
        theta = self.ReArea(theta)
        return self.fixed_RZ_gate % (bit, theta)

    def P_gate(self, theta, bit):
        theta = self.ReArea(theta)
        return self.fixed_RZ_gate % (bit, theta)

    def U_gate(self, theta, phi, lamb, bit):
        cir = ''''''
        cir += self.P_gate(phi + np.pi, bit)
        cir += self.SX_gate(bit)
        cir += self.P_gate(theta + np.pi, bit)
        cir += self.SX_gate(bit)
        cir += self.P_gate(lamb, bit)     
        return cir

    def CX_gate(self, bit1, bit2):
        cir = ''''''
        cir += self.Y2M_gate(bit2)
        cir += self.CZ_gate(bit1, bit2)
        cir += self.Y2P_gate(bit2)
        return cir

    def CZ_gate(self, bit1, bit2):
        return self.fixed_CZ_gate % (bit1, bit2)

    def measure_gate(self, bit):
        return self.fixed_measure_gate % (bit)

    def swap_gate(self, bit1, bit2):
        cir = ''''''
        cir += self.CX_gate(bit1, bit2)
        cir += self.CX_gate(bit2, bit1)
        cir += self.CX_gate(bit1, bit2)
        return cir

    def CX_swap_gate(self, bit1, bit2, bit3):
        cir = ''''''
        cir += self.Y2M_gate(bit2)
        cir += self.swap_gate(bit2,bit3)
        cir += self.CZ_gate(bit1,bit3)
        cir += self.swap_gate(bit3,bit2)
        cir += self.Y2P_gate(bit2)
        return cir


# 量子HHL
class QHHL():
    def __init__(self):
        self.G = Qgate()

    def circuit(self):

        na = np.array([2])
        nl = np.array([9,14])
        nb = np.array([20])

        cir = ''''''
        #1.1 Initialization
        
        cir += self.G.H_gate(nb[0])
        
        #1.2 Create superposition
        for i in nl:
            cir += self.G.H_gate(i)
        # 1.3 Apply controlled-U gate
        cir += self.G.P_gate(pi/2, nl[0])
        cir += self.G.P_gate(pi, nl[1])
        cir += self.G.CX_gate(nl[1], nb[0])
        ## 1.4 Inverse quantum Fourier transform
        cir += self.G.swap_gate(nl[0] , nl[1])

        #iQFT
        cir += self.G.H_gate(nl[1])
        cir += self.G.CX_gate(nl[0] , nl[1])
        cir += self.G.RZ_gate(pi/4, nl[1])
        cir += self.G.CX_gate(nl[0] , nl[1])
        cir += self.G.RZ_gate(-pi/4, nl[1])
        cir += self.G.RZ_gate(-pi/4, nl[0])
        cir += self.G.H_gate(nl[0])
        ## 2.1 SWAP
        cir += self.G.X_gate(nl[0])
        # 2.2 rotation R_y
        cir += self.G.RY_gate(pi/16, na[0])
        cir += self.G.CX_gate(nl[0], na[0])
        cir += self.G.RY_gate(-pi/16, na[0])
        cir += self.G.CX_gate(nl[0], na[0])       

        cir += self.G.RY_gate(pi/32, na[0])
        cir += self.G.CX_swap_gate(nl[1] , na[0], nl[0])
        cir += self.G.RY_gate(-pi/32, na[0])
        cir += self.G.CX_swap_gate(nl[1] , na[0], nl[0]) 

        cir += self.G.X_gate(nl[0])
        #3.1 iQFT
        cir += self.G.H_gate(nl[0])
        cir += self.G.CX_gate(nl[0] , nl[1])
        cir += self.G.RZ_gate(-pi/4, nl[1])
        cir += self.G.CX_gate(nl[0] , nl[1])
        cir += self.G.RZ_gate(pi/4, nl[1])
        cir += self.G.RZ_gate(pi/4, nl[0])
        cir += self.G.H_gate(nl[1])

        cir += self.G.swap_gate(nl[0], nl[1])
        #3.1 additional
        cir += self.G.CX_gate(nl[1], nb[0])
        cir += self.G.RZ_gate(-pi, nl[0])
        cir += self.G.RZ_gate(-pi/2, nl[1])
        
        cir += self.G.measure_gate(na[0])
        cir += self.G.measure_gate(nb[0])

        return cir


    def solve(self):
        account = Account(login_key='****',machine_name='ClosedBetaQC')

        cir = self.circuit()
        query_id = account.submit_job(cir, num_shots=8000)
        outputstate = {}
        if query_id:
            outputstate = account.query_experiment(query_id, max_wait_time=36000)
            if outputstate == {}:
                print(cir)

        num0 = 0
        num1 = 0
        for i in outputstate["results"]:
            if i[0] == 1:
                if i[1] == 0:
                    num0 = num0 + 1
                else:
                    num1 = num1 + 1

        print("total: ", num0+num1)
        print("0: ", num0/(num0+num1))
        print("1: ", num1/(num0+num1))

Eq = QHHL()
Eq.solve()