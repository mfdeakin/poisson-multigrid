
#include <cmath>
#include <gtest/gtest.h>
#include <random>

#include "multigrid.hpp"

constexpr real pi = 3.1415926535897932;

real test_func(real x, real y) { return std::sin(pi * x) * std::sin(pi * y); }

TEST(multigrid, transfer) {
  constexpr int x_cells = 64, y_cells = 64;
  BoundaryConditions bc;
  Mesh src({0.0, 0.0}, {1.0, 1.0}, x_cells, y_cells);
  for (int i = 0; i < src.x_cells(); i++) {
    const real x = src.median_x(i);
    for (int j = 0; j < src.y_cells(); j++) {
      const real y = src.median_y(j);
      src[{i, j}] = test_func(x, y);
    }
  }
  PoissonFVMGSolver<1> dest({0.0, 0.0}, {1.0, 1.0}, x_cells / 2, y_cells / 2,
                            bc, [](real, real) { return 0.0; });
  dest.restrict(src);
  const Mesh &restricted = dest.source();
  for (int i = 0; i < restricted.x_cells(); i++) {
    const real x = restricted.median_x(i);
    for (int j = 0; j < restricted.y_cells(); j++) {
			const real y = restricted.median_y(j);
      dest[{i, j}] = restricted[{i, j}];
      EXPECT_NEAR((restricted[{i, j}]), test_func(x, y), 1e-3);
    }
  }
  Mesh result({0.0, 0.0}, {1.0, 1.0}, x_cells, y_cells);
  dest.prolongate(result);
  for (int i = 0; i < result.x_cells(); i++) {
    const real x = src.median_x(i);
    for (int j = 0; j < result.y_cells(); j++) {
      const real y = src.median_y(j);
      EXPECT_NEAR((result[{i, j}]), test_func(x, y), 0.025);
    }
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
