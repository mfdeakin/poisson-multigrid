
#include <algorithm>
#include <cmath>
#include <limits>

#include "multigrid.hpp"

Mesh::Mesh(const std::pair<real, real> &corner_1,
           const std::pair<real, real> &corner_2, const size_t cells_x,
           const size_t cells_y) noexcept
    : min_x_(std::min(corner_1.first, corner_2.first)),
      max_x_(std::max(corner_1.first, corner_2.first)),
      min_y_(std::min(corner_1.second, corner_2.second)),
      max_y_(std::max(corner_1.second, corner_2.second)),
      dx_((max_x_ - min_x_) / cells_x), dy_((max_y_ - min_y_) / cells_y),
      cva_(std::array<size_t, 2>{cells_x + 2 * ghost_cells,
                                 cells_y + 2 * ghost_cells}) {
  for (size_t i = 0; i < cva_.shape()[0]; i++) {
    for (size_t j = 0; j < cva_.shape()[1]; j++) {
      cva_(i, j) = 0.0;
    }
  }
}

Mesh::Mesh(const std::pair<real, real> &corner_1,
           const std::pair<real, real> &corner_2, const size_t cells_x,
           const size_t cells_y, std::function<real(real, real)> f) noexcept
    : Mesh(corner_1, corner_2, cells_x, cells_y) {
  for (int i = 0; i < cva_.shape()[0]; i++) {
    for (int j = 0; j < cva_.shape()[1]; j++) {
      cva_(i, j) = f(median_x(i - ghost_cells), median_y(j - ghost_cells));
    }
  }
}

// Use bilinear interpolation to estimate the value at the requested point
// Note that we can use this to implement multigrid for mesh size ratios
// other than factors of 2
real Mesh::interpolate(real x, real y) const noexcept {
  const int left_cell = x_idx(x - dx_ / 2.0);
  const int right_cell = left_cell + 1;
  const int top_cell = y_idx(y + dy_ / 2.0);
  const int bottom_cell = top_cell - 1;
  assert(left_cell >= -1);
  assert(right_cell <= cells_x());
  assert(bottom_cell >= -1);
  assert(top_cell <= cells_y());
  const real left_weight = (median_x(right_cell) - x) / dx_;
  const real bottom_weight = (median_y(top_cell) - y) / dy_;
  return left_weight *
             (bottom_weight * cv_average(left_cell, bottom_cell) +
              (1.0 - bottom_weight) * cv_average(left_cell, top_cell)) +
         (1.0 - left_weight) *
             (bottom_weight * cv_average(right_cell, bottom_cell) +
              (1.0 - bottom_weight) * cv_average(right_cell, top_cell));
}

BoundaryConditions::BoundaryConditions(
    const BC_Type left_t, const std::function<real(real, real)> &left_bc,
    const BC_Type right_t, const std::function<real(real, real)> &right_bc,
    const BC_Type top_t, const std::function<real(real, real)> &top_bc,
    const BC_Type bottom_t, const std::function<real(real, real)> &bottom_bc)
    : left_t_(left_t), right_t_(right_t), top_t_(top_t), bottom_t_(bottom_t),
      left_bc_(left_bc), right_bc_(right_bc), top_bc_(top_bc),
      bottom_bc_(bottom_bc) {}

void BoundaryConditions::apply(Mesh &mesh) const noexcept {
  const real y_top = mesh.top_y(mesh.cells_y() - 1);
  const real y_bottom = mesh.bottom_y(0);
  for (int i = 0; i < mesh.cells_x(); i++) {
    const real x = mesh.median_x(i);
    real bndry_val = mesh[{i, 0}];
    if (top_t_ == BC_Type::dirichlet) {
      bndry_val = -bndry_val;
    }
    mesh[{i, -1}] = 2.0 * bottom_bc_(x, y_bottom) + bndry_val;
  }
  for (int i = 0; i < mesh.cells_x(); i++) {
    const real x = mesh.median_x(i);
    real bndry_val = mesh[{i, mesh.cells_y() - 1}];
    if (top_t_ == BC_Type::dirichlet) {
      bndry_val = -bndry_val;
    }
    mesh[{i, mesh.cells_y()}] = 2.0 * top_bc_(x, y_top) + bndry_val;
  }

  const real x_left = mesh.left_x(0);
  const real x_right = mesh.right_x(mesh.cells_x() - 1);
  for (int j = 0; j < mesh.cells_y(); j++) {
    const real y = mesh.median_y(j);
    real bndry_val = mesh[{0, j}];
    if (left_t_ == BC_Type::dirichlet) {
      bndry_val = -bndry_val;
    }
    mesh[{-1, j}] = 2.0 * left_bc_(x_left, y) + bndry_val;

    bndry_val = mesh[{mesh.cells_x() - 1, j}];
    if (right_t_ == BC_Type::dirichlet) {
      bndry_val = -bndry_val;
    }
    mesh[{mesh.cells_x(), j}] = 2.0 * right_bc_(x_right, y) + bndry_val;
  }
}

real zero_source(real, real) { return 0.0; }

PoissonFVMGSolverBase::PoissonFVMGSolverBase(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y, const BoundaryConditions &bc) noexcept
    : Mesh(corner_1, corner_2, cells_x, cells_y), bc_(bc),
      source_(corner_1, corner_2, cells_x, cells_y) {}

PoissonFVMGSolverBase::PoissonFVMGSolverBase(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y, const BoundaryConditions &bc,
    const std::function<real(real, real)> &source) noexcept
    : Mesh(corner_1, corner_2, cells_x, cells_y), bc_(bc),
      source_(corner_1, corner_2, cells_x, cells_y, source) {}

real PoissonFVMGSolverBase::delta(const int i, const int j) const noexcept {
  // Computes the difference between the Laplacian and the source term
  assert(i >= 0);
  assert(i < cells_x());
  assert(j >= 0);
  assert(j < cells_y());
  // Our problem is the form of \del u = f
  // The residual is then just r = \del u - f
  return ((cv_average(i - 1, j) - 2.0 * cv_average(i, j) +
           cv_average(i + 1, j)) /
              (dx_ * dx_) +
          (cv_average(i, j - 1) - 2.0 * cv_average(i, j) +
           cv_average(i, j + 1)) /
              (dy_ * dy_)) -
         source_[{i, j}];
}

real PoissonFVMGSolverBase::poisson_pgs_or(const real or_term) noexcept {
  real max_diff = -std::numeric_limits<real>::infinity();
  bc_.apply(*this);
  for (int i = 0; i < cells_x(); i++) {
    for (int j = 0; j < cells_y(); j++) {
      const real diff = or_term * delta(i, j);
      max_diff = std::max(max_diff, diff);
      cv_average(i, j) += diff;
    }
  }
  return max_diff;
}

void PoissonFVMGSolverBase::restrict(
    const PoissonFVMGSolverBase &src) noexcept {
  // We're assuming the cells are uniform
  assert(src.cells_x() == 2 * cells_x());
  assert(src.cells_y() == 2 * cells_y());
  for (int i = 0; i < cells_x(); i++) {
    for (int j = 0; j < cells_y(); j++) {
      source_[{i, j}] =
          0.25 *
          (src.delta(2 * i, 2 * j) + src.delta(2 * i, 2 * j + 1) +
           src.delta(2 * i + 1, 2 * j) + src.delta(2 * i + 1, 2 * j + 1));
      // It's not clear how to best precondition the coarser grids, so just
      // assume it's of the form of the source term, as suggested by the
      // periodic solutions to the Poisson problem
      // This is slower than assuming our previous estimate to the error was
      // reasonable, but it doesn't blow up from our previous solution being a
      // bad initial guess.
      // Another possibility is to set it to 0, it appears to work just as well
      // cv_average(i, j) = source_[{i, j}];
      cv_average(i, j) = 0.0;
    }
  }
}

real PoissonFVMGSolverBase::prolongate(Mesh &dest) const noexcept {
  assert(dest.cells_x() == 2 * cells_x());
  assert(dest.cells_y() == 2 * cells_y());
  real max_diff = -std::numeric_limits<real>::infinity();
  for (int i = 0; i < dest.cells_x(); i++) {
    const real x = dest.median_x(i);
    for (int j = 0; j < dest.cells_y(); j++) {
      const real y = dest.median_y(j);
      // The less accurate alternative
      // const real diff = cv_average(i / 2, j / 2);
      const real diff = interpolate(x, y);
      max_diff = std::max(max_diff, diff);
      dest[{i, j}] += diff;
    }
  }
  return max_diff;
}

BoundaryConditions mg_bc(const BoundaryConditions &src) {
  return BoundaryConditions(
      src.left_bc().first, zero_source, src.right_bc().first, zero_source,
      src.bottom_bc().first, zero_source, src.top_bc().first, zero_source);
}

template <int mg_levels_>
PoissonFVMGSolver<mg_levels_>::PoissonFVMGSolver(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y, const BoundaryConditions &bc) noexcept
    : PoissonFVMGSolverBase(corner_1, corner_2, cells_x, cells_y, bc),
      multilev_(corner_1, corner_2, cells_x / 2, cells_y / 2, mg_bc(bc),
                zero_source) {
  assert(cells_x % (1 << (mg_levels_ - 1)) == 0);
  assert(cells_y % (1 << (mg_levels_ - 1)) == 0);
}

template <int mg_levels_>
PoissonFVMGSolver<mg_levels_>::PoissonFVMGSolver(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t cells_x,
    const size_t cells_y, const BoundaryConditions &bc,
    const std::function<real(real, real)> &source) noexcept
    : PoissonFVMGSolverBase(corner_1, corner_2, cells_x, cells_y, bc, source),
      multilev_(corner_1, corner_2, cells_x / 2, cells_y / 2, mg_bc(bc),
                zero_source) {
  assert(cells_x % (1 << (mg_levels_ - 1)) == 0);
  assert(cells_y % (1 << (mg_levels_ - 1)) == 0);
}

PoissonFVMGSolver<1>::PoissonFVMGSolver(const std::pair<real, real> &corner_1,
                                        const std::pair<real, real> &corner_2,
                                        const size_t cells_x,
                                        const size_t cells_y,
                                        const BoundaryConditions &bc) noexcept
    : PoissonFVMGSolverBase(corner_1, corner_2, cells_x, cells_y, bc) {}

PoissonFVMGSolver<1>::PoissonFVMGSolver(
    const std::pair<double, double> &corner_1,
    const std::pair<double, double> &corner_2, size_t cells_x, size_t cells_y,
    const BoundaryConditions &bc,
    const std::function<double(double, double)> &source) noexcept
    : PoissonFVMGSolverBase(corner_1, corner_2, cells_x, cells_y, bc, source) {}

// Explicitly instantiate up multilevel implementations up to 10 levels
// Note that 1 level is already instantiated because it's a specialization
template class PoissonFVMGSolver<2>;
template class PoissonFVMGSolver<3>;
template class PoissonFVMGSolver<4>;
template class PoissonFVMGSolver<5>;
template class PoissonFVMGSolver<6>;
template class PoissonFVMGSolver<7>;
template class PoissonFVMGSolver<8>;
template class PoissonFVMGSolver<9>;
template class PoissonFVMGSolver<10>;
