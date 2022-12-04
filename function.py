# plot of simple function
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import random

#z = sin(x)*cos(y)*2y*x
def Draw2DForRandomY():
    x_coordinates = []
    z_coordinates = []
    x = -5
    y = round(random.uniform(-5,5), 2)
    for i in range(1001):
        sinus = np.sin(x)    
        cosinus = np.cos(y) 
        z_coord = sinus * cosinus * (2*y) * x
        print(f'x:{x}; y:{y}; z:{z_coord}')
        x_coordinates.append(x)
        z_coordinates.append(z_coord)
        x = x + 0.01
    x = list(map(float, x_coordinates))
    z = list(map(float, z_coordinates))
    plt.plot(x,z)
    plt.show()
    
def f(x,y):
    return (np.sin(x) * np.cos(y) * (2*y) * x)*1.2

def Draw3D():
    print('plotting...')
    x = np.linspace(-10, 10, 4001)
    y = np.linspace(-10, 10, 4001)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    mycmap = plt.get_cmap('gist_earth')
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=mycmap, linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-150, 150)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
        
Draw3D()
#Draw2DForRandomY()