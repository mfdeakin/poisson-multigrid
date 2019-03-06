
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <vector>

#include "multigrid.hpp"

namespace py = pybind11;

// This just exports all the objects needed in Python

template <int mg_levels> void define_solver(py::module module) {
  std::stringstream ss;
  ss << "PoissonFVMG_" << mg_levels;
  using Solver = PoissonFVMGSolver<mg_levels>;
  py::class_<Solver, PoissonFVMGSolverBase> poisson(module, ss.str().c_str());
  poisson
      .def(
          py::init<const std::pair<real, real> &, const std::pair<real, real> &,
                   size_t, size_t, BoundaryConditions &,
                   const std::function<real(real, real)> &>())
      .def(
          py::init<const std::pair<real, real> &, const std::pair<real, real> &,
                   size_t, size_t, BoundaryConditions &>());
  using container = std::vector<std::pair<int, int>>;
  poisson.def("solve", &Solver::template solve<container>);
  if constexpr (mg_levels > 1) {
    define_solver<mg_levels - 1>(module);
  }
}

PYBIND11_MODULE(multigrid, module) {
  module.doc() = "C++ Multigrid Solver for the Poisson Equation";
  py::class_<Mesh> mesh(module, "Mesh");
  mesh.def(py::init<std::pair<real, real>, std::pair<real, real>, size_t,
                    size_t, std::function<real(real, real)>>())
      .def(py::init<std::pair<real, real>, std::pair<real, real>, size_t,
                    size_t>());
  mesh.def("__getitem__",
           [](const Mesh &m, std::pair<int, int> p) { return m[p]; })
      .def("__setitem__",
           [](Mesh &m, std::pair<int, int> p, real val) {
             m[p] = val;
             return m;
           })
      .def("cells_x", &Mesh::cells_x)
      .def("cells_y", &Mesh::cells_y)
      .def("median_x", &Mesh::median_x)
      .def("median_y", &Mesh::median_y)
      .def("array",
           [](Mesh &m) {
             return py::array((m.cells_x() + 2 * m.ghost_cells) *
                                  (m.cells_y() + 2 * m.ghost_cells),
                              m.data().data())
                 .attr("reshape")(m.cells_x() + 2 * m.ghost_cells,
                                  m.cells_y() + 2 * m.ghost_cells);
           })
      .def("interpolate", &Mesh::interpolate)
      .def("grid_x",
           [](Mesh &m) {
             xt::xtensor<real, 2> grid(std::array<size_t, 2>{
                 static_cast<size_t>(m.cells_x() + 2 * m.ghost_cells),
                 static_cast<size_t>(m.cells_y() + 2 * m.ghost_cells)});
             for (int i = 0; i < m.cells_x() + 2 * m.ghost_cells; i++) {
               for (int j = 0; j < m.cells_y() + 2 * m.ghost_cells; j++) {
                 grid(i, j) = m.median_x(i - m.ghost_cells);
               }
             }
             return py::array((m.cells_x() + 2 * m.ghost_cells) *
                                  (m.cells_y() + 2 * m.ghost_cells),
                              grid.data())
                 .attr("reshape")(m.cells_x() + 2 * m.ghost_cells,
                                  m.cells_y() + 2 * m.ghost_cells);
           })
      .def("grid_y", [](Mesh &m) {
        xt::xtensor<real, 2> grid(std::array<size_t, 2>{
            static_cast<size_t>(m.cells_x() + 2 * m.ghost_cells),
            static_cast<size_t>(m.cells_y() + 2 * m.ghost_cells)});
        for (int i = 0; i < m.cells_x() + 2 * m.ghost_cells; i++) {
          for (int j = 0; j < m.cells_y() + 2 * m.ghost_cells; j++) {
            grid(i, j) = m.median_y(j - m.ghost_cells);
          }
        }
        return py::array((m.cells_x() + 2 * m.ghost_cells) *
                             (m.cells_y() + 2 * m.ghost_cells),
                         grid.data())
            .attr("reshape")(m.cells_x() + 2 * m.ghost_cells,
                             m.cells_y() + 2 * m.ghost_cells);
      });

  py::class_<BoundaryConditions> bc(module, "BoundaryConditions");
  bc.def(py::init<BoundaryConditions::BC_Type, std::function<real(real, real)>,
                  BoundaryConditions::BC_Type, std::function<real(real, real)>,
                  BoundaryConditions::BC_Type, std::function<real(real, real)>,
                  BoundaryConditions::BC_Type,
                  std::function<real(real, real)>>())
      .def(py::init<>())
      .def("left_bc", &BoundaryConditions::left_bc)
      .def("right_bc", &BoundaryConditions::right_bc)
      .def("top_bc", &BoundaryConditions::top_bc)
      .def("bottom_bc", &BoundaryConditions::bottom_bc)
      .def("apply", &BoundaryConditions::apply);

  py::enum_<BoundaryConditions::BC_Type>(bc, "BC_Type")
      .value("dirichlet", BoundaryConditions::BC_Type::dirichlet)
      .value("neumann", BoundaryConditions::BC_Type::neumann)
      .export_values();

  using Base = PoissonFVMGSolverBase;
  py::class_<PoissonFVMGSolverBase, Mesh> poisson_base(module,
                                                       "PoissonFVMGBase");
  poisson_base
      .def(
          py::init<const std::pair<real, real> &, const std::pair<real, real> &,
                   size_t, size_t, BoundaryConditions &,
                   const std::function<real(real, real)> &>())
      .def(
          py::init<const std::pair<real, real> &, const std::pair<real, real> &,
                   size_t, size_t, BoundaryConditions &>());
  poisson_base.def("restrict", &Base::restrict)
      .def("prolongate", &Base::prolongate)
      .def("source", &Base::source)
      .def("poisson_pgs_or", &Base::poisson_pgs_or);

  define_solver<10>(module);
}
