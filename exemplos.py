import numpy as np

'''
As equaçõe diferenciais são do tipo:

d(alpha(x)U')/dx + beta(x)U' + gamma(x)U = f

x em [a,b]

e com condições de contorno

U(a) = Ua
U(b) = Ub

'''

class exemplo1(object):
    def __init__(self):
        self.alpha = lambda x: -1
        self.beta = lambda x: 0
        self.gamma = lambda x: 1
        self.f = lambda x: x
        self.a = 0
        self.b = 1
        self.ua = 0
        self.ub = 0
        self.solucao = lambda x: x - np.sinh(x)/np.sinh(1)
        self.derivada = lambda x: 1 - np.cosh(x)/np.sinh(1)
        
class exemplo2(object):
    def __init__(self):
        self.alpha = lambda x: -1
        self.beta = lambda x: 0
        self.gamma = lambda x: 1
        self.f = lambda x: 1
        self.a = 0
        self.b = 1
        self.ua = 1
        self.ub = 0
        self.solucao = lambda x: 1 - np.sinh(x)/np.sinh(1)
        self.derivada = lambda x: - np.cosh(x)/np.sinh(1)
        
class exemplo3(object):
    def __init__(self):
        self.alpha = lambda x: -1
        self.beta = lambda x: 0
        self.gamma = lambda x: 10**(5)
        self.f = lambda x: 10**(5)
        self.a = 0
        self.b = 1
        self.ua = 1
        self.ub = 0
        self.solucao = lambda x: 1 - np.sinh(((10)**(1/2)*100)*x)/np.sinh(((10)**(1/2)*100))
        self.derivada = lambda x: - ((10)**(1/2))*10**(-5)*np.cosh(((10)**(1/2))*10**(-5)*x)/np.sinh(((10)**(1/2))*10**(-5))
        
class exemplo4(object):
    def __init__(self):
        self.alpha = lambda x: x**2
        self.beta = lambda x: -5*x
        self.gamma = lambda x: 4
        self.f = lambda x: x
        self.a = 1
        self.b = 2
        self.ua = 0
        self.ub = 0
        self.solucao = lambda x: x*(-x+((x*np.log(x))/np.log(4))+1)
        self.derivada = lambda x: (2*x*np.log(x)+(1-2*np.log(4))*x+np.log(4))/np.log(4)
