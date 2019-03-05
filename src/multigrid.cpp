
#include <algorithm>
#include <cmath>
#include <limits>

#include "multigrid.hpp"

Mesh::Mesh(const std::pair<real, real> &corner_1,
           const std::pair<real, real> &corner_2, const size_t x_cells,
           const size_t y_cells) noexcept
    : min_x_(std::min(corner_1.first, corner_2.first)),
      max_x_(std::max(corner_1.first, corner_2.first)),
      min_y_(std::min(corner_1.second, corner_2.second)),
      max_y_(std::max(corner_1.second, corner_2.second)),
      dx_((max_x_ - min_x_) / x_cells), dy_((max_y_ - min_y_) / y_cells),
      cva_(std::array<size_t, 2>{x_cells + 2 * ghost_cells,
                                 y_cells + 2 * ghost_cells}) {}

Mesh::Mesh(const std::pair<real, real> &corner_1,
           const std::pair<real, real> &corner_2, const size_t x_cells,
           const size_t y_cells, std::function<real(real, real)> f) noexcept
    : Mesh(corner_1, corner_2, x_cells, y_cells) {
  for (int i = 0; i < cva_.shape()[0]; i++) {
    for (int j = 0; j < cva_.shape()[1]; j++) {
      cva_(i, j) = f(median_x(i - ghost_cells), median_y(j - ghost_cells));
    }
  }
}

// Use bilinear interpolation to estimate the value at the requested point
real Mesh::interpolate(real x, real y) const noexcept {
  const int i = x_idx(x);
  const int j = y_idx(y);
  const auto interp_cell = [=](real coord, real median, int idx, int max_idx) {
    if (coord < median) {
      if (idx > -ghost_cells) {
        return idx - 1;
      } else {
        // Nothing to do except assume a constant function here
        return idx;
      }
    } else {
      if (idx < max_idx - 1) {
        return idx + 1;
      } else {
        // Nothing to do except assume a constant function here
        return idx;
      }
    }
  };
  const int i_interp = interp_cell(x, median_x(i), i, cva_.shape()[0]);
  const int j_interp = interp_cell(y, median_y(j), j, cva_.shape()[1]);
  const real x_off = std::abs(median_x(i) - x) / dx_;
  const real y_off = std::abs(median_y(i) - y) / dy_;
  return (1.0 - x_off) * ((1.0 - y_off) * cv_average(i, j) +
                          y_off * cv_average(i, j_interp)) +
         x_off * ((1.0 - y_off) * cv_average(i_interp, j) +
                  y_off * cv_average(i_interp, j_interp));
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
  const real y_top = mesh.median_y(0);
  const real y_bottom = mesh.median_y(mesh.y_cells());
  for (int i = 0; i < mesh.x_cells(); i++) {
    const real x = mesh.median_x(i);
    real bndry_val = mesh[{i, 0}];
    if (top_t_ == BC_Type::dirichlet) {
      bndry_val = -bndry_val;
    }
    mesh[{i, -1}] = 2.0 * bottom_bc_(x, y_bottom) + bndry_val;

    bndry_val = mesh[{i, mesh.y_cells()}];
    if (top_t_ == BC_Type::dirichlet) {
      bndry_val = -bndry_val;
    }
    mesh[{i, mesh.y_cells() + 1}] = 2.0 * top_bc_(x, y_top) + bndry_val;
  }

  const real x_left = mesh.median_x(0);
  const real x_right = mesh.median_x(mesh.x_cells());
  for (int j = 0; j < mesh.y_cells(); j++) {
    const real y = mesh.median_y(j);
    real bndry_val = mesh[{0, j}];
    if (left_t_ == BC_Type::dirichlet) {
      bndry_val = -bndry_val;
    }
    mesh[{-1, j}] = 2.0 * left_bc_(x_left, y) + bndry_val;

    bndry_val = mesh[{mesh.x_cells(), j}];
    if (right_t_ == BC_Type::dirichlet) {
      bndry_val = -bndry_val;
    }
    mesh[{mesh.x_cells() + 1, j}] = 2.0 * right_bc_(x_right, y) + bndry_val;
  }
}

real zero_source(real, real) { return 0.0; }

BoundaryConditions mg_bc(BoundaryConditions &src) {
  return BoundaryConditions(
      src.left_bc().first, zero_source, src.right_bc().first, zero_source,
      src.bottom_bc().first, zero_source, src.top_bc().first, zero_source);
}

template <int mg_levels>
PoissonFVMGSolver<mg_levels>
coarsened(const std::pair<real, real> &corner_1,
          const std::pair<real, real> &corner_2, const size_t x_cells,
          const size_t y_cells, BoundaryConditions bc,
          const std::function<real(real, real)> &source) {
  assert(x_cells % 2 == 0);
  assert(y_cells % 2 == 0);
  BoundaryConditions coarse_bc(mg_bc(bc));
  PoissonFVMGSolver<mg_levels> mg(corner_1, corner_2, x_cells / 2, y_cells / 2,
                                  coarse_bc, source);
  return mg;
}

template <int mg_levels_>
PoissonFVMGSolver<mg_levels_>::PoissonFVMGSolver(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t x_cells,
    const size_t y_cells, BoundaryConditions &bc,
    const std::function<real(real, real)> &source) noexcept
    : Mesh(corner_1, corner_2, x_cells, y_cells),
      multilev_(coarsened<mg_levels_ - 1>(corner_1, corner_2, x_cells / 2,
                                          y_cells / 2, bc, zero_source)),
      bc_(bc), source_(corner_1, corner_2, x_cells, y_cells),
      delta_(corner_1, corner_2, x_cells, y_cells) {
	assert(x_cells % (1 << mg_levels_) == 0);
	assert(y_cells % (1 << mg_levels_) == 0);
  for (int i = -1; i < x_cells + 1; i++) {
    for (int j = -1; j < y_cells + 1; j++) {
      const real x = this->median_x(i);
      const real y = this->median_y(j);
      source_[{i, j}] = source(x, y);
    }
  }
}

template <int mg_levels_>
real PoissonFVMGSolver<mg_levels_>::delta(const int i, const int j) const
    noexcept {
  // Computes the difference between the Laplacian and the source term
  return ((this->cv_average(i - 1, j) + this->cv_average(i + 1, j)) +
          (this->cv_average(i, j - 1) + this->cv_average(i, j + 1))) /
             4.0 +
         source_[{i, j}] - this->cv_average(i, j);
}

template <int mg_levels_>
void PoissonFVMGSolver<mg_levels_>::residual() noexcept {
  for (int i = 0; i < this->x_cells(); i++) {
    for (int j = 0; j < this->y_cells(); j++) {
      delta_[{i, j}] = delta(i, j);
    }
  }
}

template <int mg_levels_>
real PoissonFVMGSolver<mg_levels_>::poisson_pgs_or(
    const real or_term) noexcept {
  real max_diff = -std::numeric_limits<real>::infinity();
  for (int i = 0; i < this->x_cells(); i++) {
    for (int j = 0; j < this->y_cells(); j++) {
      const real diff = or_term * delta(i, j);
      max_diff = std::max(max_diff, diff);
      this->cv_average(i, j) += or_term * diff;
    }
  }
  return max_diff;
}

template <int mg_levels_>
void PoissonFVMGSolver<mg_levels_>::restrict(const Mesh &src) noexcept {
  // We're assuming cells
  assert(src.x_cells() == 2 * this->x_cells());
  assert(src.y_cells() == 2 * this->y_cells());
  for (int i = 0; i < this->x_cells(); i++) {
    for (int j = 0; j < this->y_cells(); j++) {
      this->source_[{i, j}] =
          0.25 * (src[{2 * i, 2 * j}] + src[{2 * i, 2 * j + 1}] +
                  src[{2 * i + 1, 2 * j}] + src[{2 * i + 1, 2 * j + 1}]);
    }
  }
}

template <int mg_levels_>
real PoissonFVMGSolver<mg_levels_>::prolongate(Mesh &dest) const noexcept {
  assert(dest.x_cells() == 2 * this->x_cells());
  assert(dest.y_cells() == 2 * this->y_cells());
  real max_diff = -std::numeric_limits<real>::infinity();
  for (int i = 0; i < dest.x_cells(); i++) {
    for (int j = 0; j < dest.y_cells(); j++) {
      const real diff = this->cv_average(i / 2, j / 2);
      max_diff = std::max(max_diff, diff);
      dest[{i, j}] += diff;
    }
  }
  return max_diff;
}

// The next versions are just the versions for mg_levels=1
PoissonFVMGSolver<1>::PoissonFVMGSolver(
    const std::pair<double, double> &corner_1,
    const std::pair<double, double> &corner_2, size_t x_cells, size_t y_cells,
    BoundaryConditions &bc,
    const std::function<double(double, double)> &source) noexcept
    : Mesh(corner_1, corner_2, x_cells, y_cells), bc_(bc),
      source_(corner_1, corner_2, x_cells, y_cells),
      delta_(corner_1, corner_2, x_cells, y_cells) {
  for (int i = -1; i < x_cells + 1; i++) {
    for (int j = -1; j < y_cells + 1; j++) {
      const real x = this->median_x(i);
      const real y = this->median_y(j);
      source_[{i, j}] = source(x, y);
    }
  }
}

real PoissonFVMGSolver<1>::delta(const int i, const int j) const noexcept {
  // Computes the difference between the Laplacian and the source term
  return ((this->cv_average(i - 1, j) + this->cv_average(i + 1, j)) +
          (this->cv_average(i, j - 1) + this->cv_average(i, j + 1))) /
             4.0 +
         source_[{i, j}] - this->cv_average(i, j);
}

void PoissonFVMGSolver<1>::residual() noexcept {
  for (int i = 0; i < this->x_cells(); i++) {
    for (int j = 0; j < this->y_cells(); j++) {
      delta_[{i, j}] = delta(i, j);
    }
  }
}

real PoissonFVMGSolver<1>::poisson_pgs_or(const real or_term) noexcept {
  real max_diff = -std::numeric_limits<real>::infinity();
  for (int i = 0; i < this->x_cells(); i++) {
    for (int j = 0; j < this->y_cells(); j++) {
      const real diff = or_term * delta(i, j);
      max_diff = std::max(max_diff, diff);
      this->cv_average(i, j) += or_term * diff;
    }
  }
  return max_diff;
}

void PoissonFVMGSolver<1>::restrict(const Mesh &src) noexcept {
  // We're assuming cells
  assert(src.x_cells() == 2 * this->x_cells());
  assert(src.y_cells() == 2 * this->y_cells());
  for (int i = 0; i < this->x_cells(); i++) {
    for (int j = 0; j < this->y_cells(); j++) {
      this->source_[{i, j}] =
          0.25 * (src[{2 * i, 2 * j}] + src[{2 * i, 2 * j + 1}] +
                  src[{2 * i + 1, 2 * j}] + src[{2 * i + 1, 2 * j + 1}]);
    }
  }
}

real PoissonFVMGSolver<1>::prolongate(Mesh &dest) const noexcept {
  assert(dest.x_cells() == 2 * this->x_cells());
  assert(dest.y_cells() == 2 * this->y_cells());
  real max_diff = -std::numeric_limits<real>::infinity();
  for (int i = 0; i < dest.x_cells(); i++) {
    for (int j = 0; j < dest.y_cells(); j++) {
      const real diff = this->cv_average(i / 2, j / 2);
      max_diff = std::max(max_diff, diff);
      dest[{i, j}] += diff;
    }
  }
  return max_diff;
}

template class PoissonFVMGSolver<2>;
template class PoissonFVMGSolver<3>;
template class PoissonFVMGSolver<4>;
template class PoissonFVMGSolver<5>;
template class PoissonFVMGSolver<6>;
template class PoissonFVMGSolver<7>;
template class PoissonFVMGSolver<8>;
template class PoissonFVMGSolver<9>;
template class PoissonFVMGSolver<10>;
