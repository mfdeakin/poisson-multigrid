
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
  py::class_<Solver> poisson(module, ss.str().c_str());
  poisson.def(
      py::init<const std::pair<real, real> &, const std::pair<real, real> &,
               size_t, size_t, BoundaryConditions &,
               const std::function<real(real, real)> &>());
  poisson.def("restrict", &Solver::restrict)
      .def("prolongate", &Solver::prolongate);
  using container = std::vector<std::pair<int, int>>;
  poisson.def("solve", &Solver::template solve<container>);
  define_solver<mg_levels - 1>(module);
}

template <> void define_solver<1>(py::module module) {
  std::stringstream ss;
  ss << "PoissonFVMG_" << 1;
  using Solver = PoissonFVMGSolver<1>;
  py::class_<Solver> poisson(module, ss.str().c_str());
  poisson.def(
      py::init<const std::pair<real, real> &, const std::pair<real, real> &,
               size_t, size_t, BoundaryConditions &,
               const std::function<real(real, real)> &>());
  poisson.def("restrict", &Solver::restrict)
      .def("prolongate", &Solver::prolongate);
  using container = std::vector<std::pair<int, int>>;
  poisson.def("solve", &Solver::template solve<container>);
}

PYBIND11_MODULE(multigrid, module) {
  module.doc() = "C++ Multigrid Solver for the Poisson Equation";
  py::class_<Mesh> mesh(module, "Mesh");
  mesh.def(py::init<std::pair<real, real>, std::pair<real, real>, size_t,
                    size_t, std::function<real(real, real)>>());
  mesh.def("__getitem__",
           [](const Mesh &m, std::pair<int, int> p) { return m[p]; })
      .def("__setitem__",
           [](Mesh &m, std::pair<int, int> p, real val) {
             m[p] = val;
             return m;
           })
      .def("array",
           [](Mesh &m) {
             return py::array((m.x_cells() + 2 * m.ghost_cells) *
                                  (m.y_cells() + 2 * m.ghost_cells),
                              m.data().data())
                 .attr("reshape")(m.x_cells() + 2 * m.ghost_cells,
                                  m.y_cells() + 2 * m.ghost_cells);
           })
      .def("interpolate", &Mesh::interpolate);

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
      .def("bottom_bc", &BoundaryConditions::bottom_bc);

  py::enum_<BoundaryConditions::BC_Type>(bc, "BC_Type")
      .value("dirichlet", BoundaryConditions::BC_Type::dirichlet)
      .value("neumann", BoundaryConditions::BC_Type::neumann)
      .export_values();

  define_solver<10>(module);
}
