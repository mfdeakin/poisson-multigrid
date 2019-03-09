from multigrid import Mesh, BoundaryConditions
from multigrid import (PoissonFVMG_1, PoissonFVMG_2, PoissonFVMG_3, PoissonFVMG_4, PoissonFVMG_5,
                       PoissonFVMG_6, PoissonFVMG_7, PoissonFVMG_8, PoissonFVMG_9, PoissonFVMG_10)
from numpy import sin, pi, nan, sum, sqrt, array

from matplotlib.pyplot import figure, contour, show, clabel, title, pcolor, colorbar, semilogy, legend
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def f(x, y):
    return sin(pi * x) * sin(pi * y)

def f_lapl(x, y):
    return -2.0 * pi * pi * sin(pi * x) * sin(pi * y)

bc = BoundaryConditions(BoundaryConditions.dirichlet, f, # Left BC
                        BoundaryConditions.dirichlet, f, # Right BC
                        BoundaryConditions.dirichlet, f, # Top BC
                        BoundaryConditions.dirichlet, f) # Bottom BC

def plot_3d(x, y, a, name):
    fig = figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(x, y, a, cmap=cm.gist_heat)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title(name)

def plot_mesh(m, name):
    plot_3d(m.grid_x(), m.grid_y(), m.array(), name)

orig = PoissonFVMG_1((0.0, 0.0), (1.0, 1.0), 64, 64, bc)
for i in range(orig.cells_x()):
    x = orig.median_x(i)
    for j in range(orig.cells_y()):
        y = orig.median_y(j)
        orig[(i, j)] = f(x, y)

bc.apply(orig)

plot_mesh(orig, "Fine sin(pi * x) * sin(pi * y)")

restriction = PoissonFVMG_1((0.0, 0.0), (1.0, 1.0), 32, 32, bc)
restriction.restrict(orig)
for i in range(restriction.cells_x()):
    for j in range(restriction.cells_y()):
        restriction[(i, j)] = restriction.source()[(i, j)] / (-2.0 * pi * pi)
plot_mesh(restriction, "Restricted sin(pi * x) * sin(pi * y)")

bc.apply(restriction)

# Restriction only occurs on the internal cells,
# the ghost cells need to be excluded when examining the error
err_restricted = (restriction.array()
                  - f(restriction.grid_x(), restriction.grid_y()))[1:-1, 1:-1]
plot_3d(restriction.grid_x()[1:-1, 1:-1], restriction.grid_y()[1:-1, 1:-1],
        err_restricted, "Restriction Error")

prolongated = Mesh((0.0, 0.0), (1.0, 1.0), 64, 64)
restriction.prolongate(prolongated)
err_prolongated = (prolongated.array() - orig.array())[1:-1, 1:-1]
plot_3d(prolongated.grid_x()[1:-1, 1:-1], prolongated.grid_y()[1:-1, 1:-1],
        err_prolongated, "Prolongated Error")
figure()
pcolor(err_prolongated)
colorbar()
title("Prolongated Error")

plot_mesh(prolongated, "Prolongated sin(pi * x) * sin(pi * y)")
