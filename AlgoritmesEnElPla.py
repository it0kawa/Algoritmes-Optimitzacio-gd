import numpy as np
from math import sqrt
from numpy import asarray
from numpy import arange
from numpy.random import rand
from numpy.random import seed
from numpy import meshgrid
from matplotlib import pyplot as plt


"""
Dades
"""
# seed per generar els nombres random
seed(1)

# rang de la funcio
limits = asarray([[-250.0, 250.0], [-250.0, 250.0]])

#nombre iteracions (passes)
nombre_iteracions = 1000

#Learning Rates
LearningRate0 = 3 #Adam
LearningRate1 = 0.02 #Momentum, NAG
LearningRate2 = 5 #Adagrad

#Beta
beta0 = 0.9 #Momentum
beta1 = 0.8 #Adam
beta2 = 0.999 #Adam

#Momentum NAG
momentum4NAG = 0.3


"""
Funcions
"""
def objective(x, y):
    """
    computa el valor de z a partir de la funcio
    """
    return (np.sin(x) * np.cos(y) * (2*y) * x)#*1.2

def derivative(x, y):
    """
    derivada parcial de la funcio (d/dx, d/dy)
    """
    return asarray([2*y*np.cos(y)*(np.sin(x) + x*np.cos(x)), -2*x*np.sin(x)*(y*np.sin(y)-np.cos(y))])


def Adam(objective, derivative, limits, nombre_iteracions, LearningRate, beta1, beta2, eps=1e-8):#eps per tal de no /0
    print('\n\nAdam: ')
    solutionsAdam = list()
    #coordenades x,y de la funcio de on apareixem
    x = [-4, 2]
    zcoord = objective(x[0], x[1])
    #inicialitzem a 0 m i v
    m = [0.0 for _ in range(limits.shape[0])]
    v = [0.0 for _ in range(limits.shape[0])]
    for t in range(nombre_iteracions):
        gradient = derivative(x[0], x[1])
        #calculem un gradient per cada iteracio
        for i in range(limits.shape[0]):
            #loop de n iteracions on n es el eix x
            m[i] = beta1 * m[i] + (1.0 - beta1) * gradient[i]
            v[i] = beta2 * v[i] + (1.0 - beta2) * gradient[i]**2
            mhat = m[i] / (1.0 - beta1**(t+1))
            vhat = v[i] / (1.0 - beta2**(t+1))
            x[i] = x[i] - LearningRate * mhat / (sqrt(vhat) + eps)
        zcoord = objective(x[0], x[1])
        # recalculem els valors per la nova x i y
        solutionsAdam.append(x.copy())
        #llista amb la trajectoria
    print(f'Punt final iteracio num {t}: x = {x[0]}, y = {x[1]}, z = {zcoord}') #print coordenades x,y,z finals
    return solutionsAdam

def Momentum(objective, derivative, limits, nombre_iteracions, LearningRate, beta0):
    print('Momentum: ')
    solutionsMomentum = list()
    x = [-1, -1]
    zcoord = objective(x[0], x[1])
    m = [0.0 for _ in range(limits.shape[0])]
    v = [0.0 for _ in range(limits.shape[0])]
    for t in range(nombre_iteracions):
        gradient = derivative(x[0], x[1])
        for i in range(limits.shape[0]):
            v[i] = beta0 * v[i] + LearningRate * gradient[i]
            x[i] = x[i] - v[i]
        zcoord = objective(x[0], x[1])
        solutionsMomentum.append(x.copy())
    print(f'Punt final iteracio num {t}: x = {x[0]}, y = {x[1]}, z = {zcoord}') #print coordenades x,y,z finals
    return solutionsMomentum

def NAG(objective, derivative, limits, nombre_iteracions, LearningRateMomentum, momentum):
    print('NAG: ')
    solutionsNAG = list()
    x = [-1, -1]
    forma = limits[:, 0] + rand(len(limits)) * (limits[:, 1] - limits[:, 0])
    #coordenades x,y de la funcio de on apareixem aleatories
    zcoord = objective(x[0], x[1])
    m = [0.0 for _ in range(limits.shape[0])]
    v = [0.0 for _ in range(limits.shape[0])]
    change = [0.0 for _ in range(limits.shape[0])]
    for t in range(nombre_iteracions):
        projected = [x[i] + momentum * change[i] for i in range(forma.shape[0])]
        gradient = derivative(projected[0], projected[1])
        NewNAGsolution = list()
        for i in range(limits.shape[0]):
            change[i] = (momentum * change[i]) - LearningRateMomentum * gradient[i]
            value = x[i] + change[i]
            NewNAGsolution.append(value)
        x = asarray(NewNAGsolution)
        solutionsNAG.append(x)
        zcoord = objective(x[0], x[1])
    print(f'Punt final iteracio num {t}: x = {x[0]}, y = {x[1]}, z = {zcoord}') #print coordenades x,y,z finals
    return solutionsNAG

def Adagrad(objective, derivative, limits, nombre_iteracions, LearningRate2):
    print('Adagrad')
    Adagradsolution = list()
    forma = limits[:, 0] + rand(len(limits)) * (limits[:, 1] - limits[:, 0])
    x = [-0.000035, -0.000022]
    sq_grad_sums = [0.0 for _ in range(limits.shape[0])]
    #variable suma de les derivades parcials al quadrat (inicialitzem al 0)
    for t in range(nombre_iteracions):
        gradient = derivative(x[0], x[1])
        for i in range(gradient.shape[0]):
            sq_grad_sums[i] += gradient[i]**2.0
        #suma de les derivades parcials al quadrat + gradient
        NewAdagradsolution = list()
        for i in range(forma.shape[0]):
            LearningRateAdagrad = LearningRate2 / (1e-8 + sqrt(sq_grad_sums[i]))
            newPos = x[i] - LearningRateAdagrad * gradient[i]
            NewAdagradsolution.append(newPos)
        x = asarray(NewAdagradsolution)
        Adagradsolution.append(x)
        #evaluate candidate point
        zcoord = objective(x[0], x[1])
    print(f'Punt final iteracio num {t}: x = {x[0]}, y = {x[1]}, z = {zcoord}') #print coordenades x,y,z finals
    return Adagradsolution


"""
Cridem les funcions
"""

#fem Adam
solutionsAdam = Adam(objective, derivative, limits, nombre_iteracions, LearningRate0, beta1, beta2)
#fem momentum
solutionsMomentum = Momentum(objective, derivative, limits, nombre_iteracions, LearningRate1, beta0)
#fem NAG
solutionsNAG = NAG(objective, derivative, limits, nombre_iteracions, LearningRate1, momentum4NAG)
#Fem Adagrad
solutionsAdagrad = Adagrad(objective, derivative, limits, nombre_iteracions, LearningRate2)


"""
Representem els algoritmes
"""

# plot Adam (cercles negres)
solutionsAdam = asarray(solutionsAdam)
plt.plot(solutionsAdam[:, 0], solutionsAdam[:, 1], '.-', color='black', label="Adam")
plt.plot(solutionsAdam[:, 0][-1], solutionsAdam[:, 1][-1], '.', color='red', label='stop') #P final vermell
plt.plot(solutionsAdam[:, 0][0], solutionsAdam[:, 1][0], '.', color='green', label='start') #P inici verd
plt.plot(linestyle='--',color='black', label='Algoritme Adam')

# plot Momentum (cercles blancs)
solutionsMomentum = asarray(solutionsMomentum)
plt.plot(solutionsMomentum[:, 0], solutionsMomentum[:, 1], '.-', color='peru', label="Momentum")
plt.plot(solutionsMomentum[:, 0][-1], solutionsMomentum[:, 1][-1], '.-', color='red') #P final vermell
plt.plot(solutionsMomentum[:, 0][0], solutionsMomentum[:, 1][0], '.-', color='green') #P inici verd
plt.plot(linestyle='--',color='brown', label='Algoritme Momentum')

# plot NAG (cercles blue)
solutionsNAG = asarray(solutionsNAG)
plt.plot(solutionsNAG[:, 0], solutionsNAG[:, 1], '.-', color='gray', label="NAG")
plt.plot(solutionsNAG[:, 0][-1], solutionsNAG[:, 1][-1], '.-', color='red') #P final vermell
plt.plot(solutionsNAG[:, 0][0], solutionsNAG[:, 1][0], '.-', color='green') #P inici verd
plt.plot(linestyle='--',color='blue', label='Algoritme NAG')

#plot Adagrad
solutionsAdagrad = asarray(solutionsAdagrad)
plt.plot(solutionsAdagrad[:, 0], solutionsAdagrad[:, 1], '.-', color='mediumorchid', label="Adagrad")
plt.plot(solutionsAdagrad[:, 0][-1], solutionsAdagrad[:, 1][-1], '.-', color='red') #P final vermell
plt.plot(solutionsAdagrad[:, 0][0], solutionsAdagrad[:, 1][0], '.-', color='green') #P inici verd
plt.plot(linestyle='--',color='white', label='Algoritme Adagrad')


"""
Muntem el plot
"""
xaxis = arange(limits[0,0], limits[0,1], 0.1)
yaxis = arange(limits[1,0], limits[1,1], 0.1)

x, y = meshgrid(xaxis, yaxis)
# crees el plot fent com l'area dels costats x, y --> (x*y)

results = objective(x, y)
# calculem la funcio

cf = plt.contourf(x, y, results, levels=50, cmap='jet')
plt.colorbar(cf)
# fer el contour plot

plt.title("Algoritmes Adam, Momentum, Nag i Adagrad a la funció: sin(x)cos(y)*x*2y")
plt.xlabel('eix x')
plt.ylabel('eix y')

plt.legend(loc = "upper left")

plt.grid()


"""
finestra amb plot
"""
plt.show()