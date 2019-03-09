
#include <cmath>
#include <gtest/gtest.h>
#include <random>

#include "multigrid.hpp"

constexpr real pi = 3.1415926535897932;

real test_func(real x, real y) { return std::sin(pi * x) * std::sin(pi * y); }
real test_residual(real x, real y) { return -2.0 * pi * pi * test_func(x, y); }

TEST(poisson, residual) {
  constexpr int cells_x = 32, cells_y = 32;
  BoundaryConditions bc; // Default homogeneous Dirichlet

  // The solver for the homogeneous Poisson problem
  PoissonFVMGSolverBase test({0.0, 0.0}, {1.0, 1.0}, cells_x, cells_y, bc);
  // Initialize its solution guess with the test function
  for (int i = 0; i < test.cells_x(); i++) {
    const real x = test.median_x(i);
    for (int j = 0; j < test.cells_y(); j++) {
      const real y = test.median_y(j);
      test[{i, j}] = test_func(x, y);
    }
  }
  for (int i = 1; i < test.cells_x() - 1; i++) {
    const real x = test.median_x(i);
    for (int j = 1; j < test.cells_y() - 1; j++) {
      const real y = test.median_y(j);
      EXPECT_NEAR(test.delta(i, j), test_residual(x, y), 2e-2);
    }
  }
}

TEST(multigrid, transfer) {
  constexpr int cells_x = 64, cells_y = 64;
  BoundaryConditions bc; // Default homogeneous Dirichlet

  // The solver for the homogeneous Poisson problem
  PoissonFVMGSolverBase src({0.0, 0.0}, {1.0, 1.0}, cells_x, cells_y, bc);
  // Initialize its solution guess and ghost cells with the test function
  for (int i = -1; i <= src.cells_x(); i++) {
    const real x = src.median_x(i);
    for (int j = -1; j <= src.cells_y(); j++) {
      const real y = src.median_y(j);
      src[{i, j}] = test_func(x, y);
    }
  }
  // src.delta(i, j) = \del test_func(x, y)
  // Exclude the boundaries from this test - we didn't apply them
  PoissonFVMGSolver<1> dest({0.0, 0.0}, {1.0, 1.0}, cells_x / 2, cells_y / 2,
                            bc);
  dest.restrict(src);
  const Mesh &restricted = dest.source();
  for (int i = 0; i < restricted.cells_x(); i++) {
    const real x = restricted.median_x(i);
    for (int j = 0; j < restricted.cells_y(); j++) {
      const real y = restricted.median_y(j);
      dest[{i, j}] = test_func(x, y);
      // Note that this error is the compounded error of the second derivative
      // approximation and the error from the transfer
      // The error is within the same bounds as the second derivative
      // approximation on a mesh of the same size, suggesting it's correct
      EXPECT_NEAR((restricted[{i, j}]), test_residual(x, y), 2e-2);
    }
  }
  Mesh result({0.0, 0.0}, {1.0, 1.0}, cells_x, cells_y);
  dest.prolongate(result);
  for (int i = 0; i < result.cells_x(); i++) {
    const real x = src.median_x(i);
    for (int j = 0; j < result.cells_y(); j++) {
      const real y = src.median_y(j);
      EXPECT_NEAR((result[{i, j}]), test_func(x, y), 2e-2);
    }
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
