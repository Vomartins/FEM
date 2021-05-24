import numpy as np
import numpy.polynomial.legendre as Lg
from sympy import *

x, y, z = symbols('x y z')

class ElementoLinear(object):

    def __init__(self, index, xA, xB, iA):
        self.num_nos = 2
        self.index = index
        self.xA = xA
        self.index_A = iA
        self.xB = xB
        self.h = xB - xA

    def phi(self):

        #Transformação de coordenadas
        #x = self.xA + (0.5*self.h)*(XI+1)
        
        X = np.array([-1,1])
        YA = np.identity(self.num_nos)[0,:]
        YB = np.identity(self.num_nos)[1,:]
        
        base = [lagrange(self.num_nos-1,X,YA), lagrange(self.num_nos-1,X,YB)]

        return base

    def dphi(self):

        #Transformação de coordenadas
        #x = self.xA + (0.5*self.h)*(x+1)
        
        base = self.phi()

        base_deriv = [lambdify(x,diff(base[0](x),x)), lambdify(x,diff(base[1](x),x))]

        return base_deriv

class ElementoQuad(object):
    def __init__(self, index, xA, xB, xC, iA):
        self.num_nos = 3
        self.index = index
        self.index_A = iA
        self.xA = xA
        self.xB = xB
        self.xC = xC
        self.h = xC - xA

    def phi(self):
        
        #Transformação de coordenadas
        #x = self.xA + (0.5*self.h)*(x+1)

        X = np.array([-1,0,1])
        YA = np.identity(self.num_nos)[0,:]
        YB = np.identity(self.num_nos)[1,:]
        YC = np.identity(self.num_nos)[2,:]

        base = [lagrange(self.num_nos-1,X,YA), lagrange(self.num_nos-1,X,YB), lagrange(self.num_nos-1,X,YC)]

        return base

    def dphi(self):

        #Transformação de coordenadas
        #x = self.xA + (0.5*self.h)*(x+1)
        
        base = self.phi()

        base_deriv = [lambdify(x,diff(base[0](x),x)), lambdify(x,diff(base[1](x),x)), lambdify(x,diff(base[2](x),x))]
        
        return base_deriv

class ElementoCubico(object):
    def __init__(self, index, xA, xB, xC, xD, iA):
        self.num_nos = 4
        self.index = index
        self.index_A = iA
        self.xA = xA
        self.xB = xB
        self.xC = xC
        self.xD = xD
        self.h = xD - xA

    def phi(self):

        #Transformação de coordenadas
        #x = self.xA + (0.5*self.h)*(x+1)

        X = np.array([-1,-1/3,1/3,1])
        YA = np.identity(self.num_nos)[0,:]
        YB = np.identity(self.num_nos)[1,:]
        YC = np.identity(self.num_nos)[2,:]
        YD = np.identity(self.num_nos)[3,:]
        
        base = [lagrange(self.num_nos-1,X,YA), lagrange(self.num_nos-1,X,YB), lagrange(self.num_nos-1,X,YC), lagrange(self.num_nos-1,X,YD)]

        return base

    def dphi(self):

        #Transformação de coordenadas
        #x = self.xA + (0.5*self.h)*(x+1)

        base = self.phi()

        base_deriv = [lambdify(x,diff(base[0](x),x)), lambdify(x,diff(base[1](x),x)), lambdify(x,diff(base[2](x),x)), lambdify(x,diff(base[3](x),x))]
        
        return base_deriv
    
'''
Funções de integração numérica
'''

def GQ_Legendre(n,a,b,f):
    aux = np.zeros(n)
    aux[-1]=1
    L = Lg.Legendre(aux)
    DL = L.deriv()
    rts = Lg.legroots(aux)
    w = 2 /((1 -(rts)**2) *((DL(rts))**2))
    pts = ((b-a)/2) *rts +((b+a)/2)
    return ((b-a)/2)*sum(w * f(pts))

def GQ_Legendre_padrao(n,f):
    a=-1
    b=1
    aux = np.zeros(n)
    aux[-1]=1
    L = Lg.Legendre(aux)
    DL = L.deriv()
    rts = Lg.legroots(aux)
    w = 2 /((1 -(rts)**2) *((DL(rts))**2))
    return sum(w * f(rts))
 
'''
Funções de interpolação numérica
'''
    
def lagrange(n, X, Y):
    def L(x,i):
        L = 1
        for j in range(n+1):
            if j != i:
                L = L*((x - X[j])/(X[i] - X[j]))
        return L
    def p(x):
        P = 0
        for i in range(n+1):
            P += Y[i]*L(x,i)
        return P
    def erro(x):
        return 0
    
    if np.size(X) != np.size(Y) or np.size(X) < n+1 or np.size(Y) < n+1:
        return erro
    else:
        return p
        
def newton(n,X,Y):
    def N(x,i):
        N = 1
        if i == 0:
            return N
        else:
            for j in range(i):
                N = N*(x - X[j])
            return N
    def DD(i,j,k):
        if i == 0:
            return Y[j]
        else:
            f = (DD(i-1,j+1,k)-DD(i-1,j,k-1))/(X[k]-X[j])
            return f
    def p(x):
        P = 0
        for i in range(n+1):
            P += DD(i,0,i)*N(x,i)
        return P
    def erro(x):
        return 0
    
    if np.size(X) != np.size(Y) or np.size(X) < n+1 or np.size(Y) < n+1:
        return erro
    else:
        return p

'''
As equaçõe diferenciais são do tipo:

d(alpha(x)U')/dx + beta(x)U' + gamma(x)U = f

x em [a,b]

e com condições de contorno

U(a) = Ua
U(b) = Ub

'''

def fem1D(n, alpha, beta, gamma, f, a, b, u0, uN, X, quad=[], cubic=[]):
    
    def local(elemento, f, ordem):
        
        def integrando(x,f,df,g,dg,alpha,beta,gamma,h):
            return gamma(elemento.xA + (0.5*h)*(x+1))*f(x)*g(x) + (2/h)*beta(elemento.xA + (0.5*h)*(x+1))*f(x)*dg(x) - (4/(h**2))*alpha(elemento.xA + (0.5*h)*(x+1))*df(x)*dg(x)

        h = elemento.h
        
        F = lambda x: f(elemento.xA + (0.5*h)*(x+1))

        if ordem == 1:
            K = np.zeros((2,2))
            fl = np.zeros(2)

            base = elemento.phi()
            base_deriv = elemento.dphi()

            for r in range(2):
                for s in range(2):
                    K[r,s] = (h/2)*GQ_Legendre_padrao(2,lambda x: integrando(x,base[r],base_deriv[r],base[s],base_deriv[s],alpha,beta,gamma,h))
            for r in range(2):
                fl[r] = (h/2)*GQ_Legendre_padrao(2,lambda x: F(x)*base[r](x))

        elif ordem == 2:
            K = np.zeros((3,3))
            fl = np.zeros(3)

            base = elemento.phi()
            base_deriv = elemento.dphi()

            for r in range(3):
                for s in range(3):
                    K[r,s] = (h/2)*GQ_Legendre_padrao(3,lambda x: integrando(x,base[r],base_deriv[r],base[s],base_deriv[s],alpha,beta,gamma,h))
            for r in range(3):
                fl[r] = (h/2)*GQ_Legendre_padrao(3,lambda x: F(x)*base[r](x))

        elif ordem == 3:
            K = np.zeros((4,4))
            fl = np.zeros(4)

            base = elemento.phi()
            base_deriv = elemento.dphi()

            for r in range(4):
                for s in range(4):
                    K[r,s] = (h/2)*GQ_Legendre_padrao(4,lambda x: integrando(x,base[r],base_deriv[r],base[s],base_deriv[s],alpha,beta,gamma,h))
            for r in range(4):
                fl[r] = (h/2)*GQ_Legendre_padrao(4,lambda x: F(x)*base[r](x))        
        else:
            raise ValueError('Invalid order value')

        return K, fl

    element_order = np.ones(n)
    element_order[quad] = 2
    element_order[cubic] = 3

    Elements = dict()
    matrix_dict = dict()
    count = 0

    for i in range(n):

        if element_order[i] == 1:
            Elements[i] = ElementoLinear(i,X[count],X[count+1],count)
            matrix_dict[i] = (local(Elements[i], f, 1))

            count = count + 1

        elif element_order[i] == 2:
            Elements[i] = ElementoQuad(i,X[count],X[count+1],X[count+2],count)
            matrix_dict[i] = (local(Elements[i], f, 2))

            count = count + 2

        elif element_order[i] == 3:
            Elements[i] = ElementoCubico(i,X[count],X[count+1],X[count+2],X[count+3],count)
            matrix_dict[i] = (local(Elements[i], f, 3))

            count = count + 3
        else:
            raise ValueError('Invalid Order value')
    
    count = count + 1

    K = np.zeros((count,count))
    F = np.zeros(count)

    for i in range(n):
        r = Elements[i].index_A
        if element_order[i] == 1:
            K[r:r+2,r:r+2] += matrix_dict[i][0]
            F[r:r+2] += matrix_dict[i][1]
        elif element_order[i] == 2:
            K[r:r+3,r:r+3] += matrix_dict[i][0]
            F[r:r+3] += matrix_dict[i][1]
        elif element_order[i] == 3:
            K[r:r+4,r:r+4] += matrix_dict[i][0]
            F[r:r+4] += matrix_dict[i][1]

    
    F = F - K[:,0]*u0 - K[:,-1]*uN
    F[0] = u0
    F[-1] = uN
    K[0,:] = 0
    K[:,0] = 0
    K[0,0] = 1
    K[-1,:] = 0
    K[:,-1] = 0
    K[-1,-1] = 1
    
    U = np.linalg.solve(K,F)

    return U, Elements

'''
Função que gera a solução numérica a partir da solução dos nós obtida com o FEM
'''

def build_solution(U, Elements):

    z = []
    sol_num = []
    sol_diff = []

    for e in Elements.values():
        iA = e.index_A

        if e.num_nos ==2:  
            aux1 = np.array([e.xA, e.xB])
            aux2 = np.array([U[iA], U[iA+1]])
            pol = np.polyfit(aux1, aux2, 1)
            x_pol = np.linspace(e.xA, e.xB, 20)
            z = np.append(z, x_pol)
            sol_num = np.append(sol_num, pol[0]*x_pol + pol[1])
            sol_diff = np.append(sol_diff, pol[0]*np.ones(len(x_pol)))

        elif e.num_nos == 3:
            aux1 = np.array([e.xA, e.xB, e.xC])
            aux2 = np.array([U[iA], U[iA+1], U[iA+2]])
            pol = np.polyfit(aux1, aux2, 2)
            x_pol = np.linspace(e.xA, e.xC, 40)
            z = np.append(z, x_pol)
            sol_num = np.append(sol_num, pol[0]*x_pol**2 + pol[1]*x_pol + pol[2])
            sol_diff = np.append(sol_diff, pol[0]*2*x_pol + pol[1])

        elif e.num_nos == 4:
            aux1 = np.array([e.xA, e.xB, e.xC, e.xD])
            aux2 = np.array([U[iA], U[iA+1], U[iA+2], U[iA+3]])
            pol = np.polyfit(aux1, aux2, 3)
            x_pol = np.linspace(e.xA, e.xD, 40)
            z = np.append(z, x_pol)
            sol_num = np.append(sol_num, pol[0]*x_pol**3 + pol[1]*x_pol**2 + pol[2]*x_pol + pol[3])
            sol_diff = np.append(sol_diff, pol[0]*3*x_pol**2 + pol[1]*2*x_pol + pol[2])

    return sol_num, sol_diff, z

'''
Função que determina os erros da solução e da derivada
'''

def erro_L2(U,u,dudx,Elements):
    z = []
    sol_num = []
    sol_diff = []
    erro_l2 = 0
    derro_l2 = 0

    for e in Elements.values():
        iA = e.index_A

        if e.num_nos ==2:  
            aux1 = np.array([e.xA, e.xB])
            aux2 = np.array([U[iA], U[iA+1]])
            pol = np.polyfit(aux1, aux2, 1)
            erro_l2 = erro_l2 + GQ_Legendre(5, e.xA, e.xB, lambda x: (u(x) - (pol[0]*x + pol[1]))**2)
            derro_l2 = derro_l2 + GQ_Legendre(5, e.xA, e.xB, lambda x: (dudx(x) -  pol[0])**2)

        elif e.num_nos == 3:
            aux1 = np.array([e.xA, e.xB, e.xC])
            aux2 = np.array([U[iA], U[iA+1], U[iA+2]])
            pol = np.polyfit(aux1, aux2, 2)
            erro_l2 = erro_l2 + GQ_Legendre(5, e.xA, e.xC, lambda x: (u(x) - (pol[0]*x**2 + pol[1]*x + pol[2]))**2)
            derro_l2 = derro_l2 + GQ_Legendre(5, e.xA, e.xC, lambda x: (dudx(x) -(pol[0]*2*x + pol[1]))**2)

        elif e.num_nos == 4:
            aux1 = np.array([e.xA, e.xB, e.xC, e.xD])
            aux2 = np.array([U[iA], U[iA+1], U[iA+2], U[iA+3]])
            pol = np.polyfit(aux1, aux2, 3)
            erro_l2 = erro_l2 + GQ_Legendre(5, e.xA, e.xD, lambda x: (u(x)- (pol[0]*x**3 + pol[1]*x**2 + pol[2]*x + pol[3]))**2)
            derro_l2 = derro_l2 + GQ_Legendre(5, e.xA, e.xD, lambda x: (dudx(x)- (pol[0]*3*x**2 + pol[1]*2*x + pol[2]))**2)

    erro_l2 = erro_l2**0.5
    derro_l2 = derro_l2**0.5

    return erro_l2, derro_l2