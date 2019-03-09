from multigrid import Mesh, BoundaryConditions
from multigrid import (PoissonFVMG_1, PoissonFVMG_2, PoissonFVMG_3, PoissonFVMG_4, PoissonFVMG_5,
                       PoissonFVMG_6, PoissonFVMG_7, PoissonFVMG_8, PoissonFVMG_9, PoissonFVMG_10)
from numpy import sin, cos, pi, nan, sum, sqrt, array

from matplotlib.pyplot import figure, contour, show, clabel, title, pcolor, colorbar, semilogy, legend
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time

def plot_3d(x, y, a, name):
    fig = figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(x, y, a, cmap=cm.gist_heat)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title(name)

def int_slice(arr):
    return arr[1:-1, 1:-1]

def plot_mesh(m, name):
    plot_3d(m.grid_x(), m.grid_y(), m.array(), name)

def f(x, y):
    return sin(pi * x) * cos(pi * y)

def f_dx(x, y):
    return pi * cos(pi * x) * cos(pi * y)

def f_dy(x, y):
    return -pi * sin(pi * x) * sin(pi * y)

def f_lapl(x, y):
    return -2.0 * pi * pi * f(x, y)

# Note that when applying neumann boundary conditions, we need the normal derivative,
# not the derivative wrt x and y - hence the negative signs for the left and bottom bc
bc = BoundaryConditions(BoundaryConditions.neumann, lambda x, y: -f_dx(x, y), # Left BC
                        BoundaryConditions.neumann, f_dx, # Right BC
                        BoundaryConditions.neumann, f_dy, # Top BC
                        BoundaryConditions.neumann, lambda x, y: -f_dy(x, y)  # Bottom BC
)

solver = PoissonFVMG_1((0.0, 0.5), (5.0,  1.5), 256, 64, bc, f_lapl)

sol = int_slice(f(solver.grid_x(), solver.grid_y()))
l2 = sqrt(sum((sol - int_slice(solver.array())) ** 2))

i = 0
while l2 > 1e-9 and i < 400:
    solver.solve([(1, 1, 1.5)])
    l2 = sqrt(sum((sol - int_slice(solver.array())) ** 2))
    i += 1

print("Completed in {} iterations, {}".format(i, l2))
plot_mesh(solver, "Solution at {} iterations".format(i))
err = int_slice(solver.array()) - sol
plot_3d(int_slice(solver.grid_x()), int_slice(solver.grid_y()),
        err, "Error")

solver = PoissonFVMG_3((0.0, 0.5), (5.0,  1.5), 256, 64, bc, f_lapl)
i = 0
while l2 > 1e-9 and i < 400:
    solver.solve([(3, 1, 2.0 / 3.0),
                  (2, 2, 2.0 / 3.0),
                  (1, 16, 1.5),
                  (2, 2, 2.0 / 3.0),
                  (3, 1, 2.0 / 3.0),])
    l2 = sqrt(sum((sol - int_slice(solver.array())) ** 2))
    i += 1

print("Completed in {} iterations, {}".format(i, l2))
plot_mesh(solver, "MG Solution at {} iterations".format(i))
err_mg = int_slice(solver.array()) - sol
plot_3d(int_slice(solver.grid_x()), int_slice(solver.grid_y()),
        err_mg, "MG Error")

plot_3d(int_slice(solver.grid_x()), int_slice(solver.grid_y()),
        err - err_mg, "Error Difference")
show()
