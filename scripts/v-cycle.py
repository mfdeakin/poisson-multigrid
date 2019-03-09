from multigrid import Mesh, BoundaryConditions
from multigrid import (PoissonFVMG_1, PoissonFVMG_2, PoissonFVMG_3, PoissonFVMG_4, PoissonFVMG_5,
                       PoissonFVMG_6, PoissonFVMG_7, PoissonFVMG_8, PoissonFVMG_9, PoissonFVMG_10)
from numpy import sin, pi, nan, sum, sqrt, array

from matplotlib.pyplot import figure, contour, show, clabel, title, pcolor, colorbar, semilogy, legend
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time

def f(x, y):
    return sin(pi * x) * sin(pi * y)

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

bc = BoundaryConditions()

# The 5 specifies the number of levels available to the multigrid
solver = PoissonFVMG_5((0.0, 0.0), (1.0,  1.0), 64, 64, bc)

def convergence_study(mg_cycles, plot_title):
    """
    mg_cycles is a list of parameters to use for each type of cycle to implement
    mg_passes is the important one,, it is a list of triples for the multigrid parameters
    The first item in the triple is the granularity to run at (higher being a finer granularity),
    ranging from 1 to the number of levels in the multigrid solver.
    The second item is the number of smoothing passes to run at that level
    The third item is the over relaxation term to use

    plot_title is the title used for the plot
    """
    for cycle in mg_cycles:
        for i in range(solver.cells_x()):
            x = solver.median_x(i)
            for j in range(solver.cells_y()):
                y = solver.median_y(j)
                solver[(i, j)] = f(x, y)

        start_time = time.clock()

        l2 = sqrt(sum(solver.array() ** 2))
        i = 0
        cycle["l2_norms"].append(l2)
        while l2 > 1e-9:
            solver.solve(cycle["mg_passes"])
            l2 = sqrt(sum(solver.array() ** 2))
            cycle["l2_norms"].append(l2)
            i += 1
            if l2 > 100.0:
                print("Abort, l2 blowing up")
                break

        end_time = time.clock()
        print("{} final iteration {}; completed in {:.6} s".format(cycle["name"], i,
                                                                   end_time - start_time))

    fig = figure()

    for cycle in mg_cycles:
        semilogy(cycle["l2_norms"], label=cycle["name"])

    title(plot_title)
    legend()

def compare_levels(or_term, plot_title):
    level_comp_cycles = [
        {"mg_passes": [(5, 1, or_term),
                       # The finer pass mostly only damped the high-frequencies,
                       # which are inaccessible at the coarser scales.
                       # So the coarser scale should damp all frequencies available to it,
                       # so using something close to the optimal over-relaxation parameter
                       # should improve convergence rates
                       (4, 2, 1.0),
                       (5, 1, or_term)],
         "name": "2 Levels",
         "l2_norms": []},
        {"mg_passes": [(5, 1, or_term),
                       (4, 1, 1.0),
                       (3, 2, 1.0),
                       (4, 1, 1.0),
                       (5, 1, or_term)],
         "name": "3 Levels",
         "l2_norms": []},
        {"mg_passes": [(5, 1, or_term),
                       (4, 1, 1.0),
                       (3, 1, 1.0),
                       (2, 2, 1.0),
                       (3, 1, 1.0),
                       (4, 1, 1.0),
                       (5, 1, or_term)],
         "name": "4 Levels",
         "l2_norms": []},
        {"mg_passes": [(5, 1, or_term),
                       (4, 1, 1.0),
                       (3, 1, 1.0),
                       (2, 1, 1.0),
                       (1, 2, 1.0),
                       (2, 1, 1.0),
                       (3, 1, 1.0),
                       (4, 1, 1.0),
                       (5, 1, or_term)],
         "name": "5 Levels",
         "l2_norms": []},
    ]

    convergence_study(level_comp_cycles, plot_title)

or_terms = [0.5, 0.6, 2.0 / 3.0, 0.8, 1.0, 1.25, 1.5]
for or_term in or_terms:
    print("or_term: {}".format(or_term))
    compare_levels(or_term, "Over Relaxation = {:.3}".format(or_term))

def compare_levels_opt_2(plot_title):
    opt_2_pass_or = (12.0 + array([sqrt(8), -sqrt(8)])) / 17.0
    level_comp_cycles = [
        {"mg_passes": [(5, 1, opt_2_pass_or[0]),
                       (5, 1, opt_2_pass_or[1]),
                       # The finer pass mostly only damped the high-frequencies,
                       # which are inaccessible at the coarser scales.
                       # So the coarser scale should damp all frequencies available to it,
                       # so using something close to the optimal over-relaxation parameter
                       # should improve convergence rates
                       (4, 2, 1.0),
                       (5, 1, opt_2_pass_or[0]),
                       (5, 1, opt_2_pass_or[1])],
         "name": "2 Levels",
         "l2_norms": []},
        {"mg_passes": [(5, 1, opt_2_pass_or[0]),
                       (5, 1, opt_2_pass_or[1]),
                       (4, 1, 1.0),
                       (3, 2, 1.0),
                       (4, 1, 1.0),
                       (5, 1, opt_2_pass_or[0]),
                       (5, 1, opt_2_pass_or[1])],
         "name": "3 Levels",
         "l2_norms": []},
        {"mg_passes": [(5, 1, opt_2_pass_or[0]),
                       (5, 1, opt_2_pass_or[1]),
                       (4, 1, 1.0),
                       (3, 1, 1.0),
                       (2, 2, 1.0),
                       (3, 1, 1.0),
                       (4, 1, 1.0),
                       (5, 1, opt_2_pass_or[0]),
                       (5, 1, opt_2_pass_or[1])],
         "name": "4 Levels",
         "l2_norms": []},
        {"mg_passes": [(5, 1, opt_2_pass_or[0]),
                       (5, 1, opt_2_pass_or[1]),
                       (4, 1, 1.0),
                       (3, 1, 1.0),
                       (2, 1, 1.0),
                       (1, 2, 1.0),
                       (2, 1, 1.0),
                       (3, 1, 1.0),
                       (4, 1, 1.0),
                       (5, 1, opt_2_pass_or[0]),
                       (5, 1, opt_2_pass_or[1])],
         "name": "5 Levels",
         "l2_norms": []},
    ]

    convergence_study(level_comp_cycles, plot_title)

compare_levels_opt_2("Optimal 2 Pass Convergence")

show()
