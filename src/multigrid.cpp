
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

void BoundaryConditions::apply(Mesh &mesh) const noexcept {
  auto apply_bc = [&](const std::pair<int, int> &idx, const real weight) {
    const auto [i_int, j_int] = idx;
    const auto [i_gc, j_gc] = adjacent(idx, mesh.x_cells(), mesh.y_cells());
    // This only works for regularly spaced meshes...
    // But we already assume that elsewhere, so okay
    const real boundary_x = (mesh.median_x(i_int) + mesh.median_x(i_gc)) / 2.0;
    const real boundary_y = (mesh.median_y(i_int) + mesh.median_y(i_gc)) / 2.0;
    mesh[{i_gc, j_gc}] =
        2.0 * bc_(boundary_x, boundary_y) + weight * mesh[{i_int, j_int}];
  };
  for (auto idx : dirichlet_) {
    apply_bc(idx, -1.0);
  }
  for (auto idx : neumann_) {
    apply_bc(idx, 1.0);
  }
}

std::pair<int, int>
BoundaryConditions::adjacent(const std::pair<int, int> &ghost_idx, int max_x,
                             int max_y) const noexcept {
  auto [i, j] = ghost_idx;
  if (i == 0) {
    return {i - 1, j};
  }
  if (i == max_x) {
    return {i + 1, j};
  }
  if (j == 0) {
    return {i, j - 1};
  }
  if (j == max_y) {
    return {i, j + 1};
  }
  // Bad indices passed
  assert(false);
  // Hopefully this will cause an issue further down the line and convince
  // the user to run in debug mode...
  return {-255, -255};
}

template <int mg_levels_>
PoissonFVMGSolver<mg_levels_>::PoissonFVMGSolver(
    const std::pair<real, real> &corner_1,
    const std::pair<real, real> &corner_2, const size_t x_cells,
    const size_t y_cells,
    const std::function<real(real, real)> &source) noexcept
    : Mesh(corner_1, corner_2, x_cells, y_cells), multilev_(),
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

template <int mg_levels_>
real PoissonFVMGSolver<mg_levels_>::delta(const int i, const int j) const
    noexcept {
  // Computes the difference between the Laplacian and the source term
  return ((this->cv_average[{i - 1, j}] + this->cv_average[{i + 1, j}]) +
          (this->cv_average[{i, j - 1}] + this->cv_average[{i, j + 1}])) /
             4.0 +
         source_[{i, j}] - this->cv_average[{i, j}];
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
real PoissonFVMGSolver<mg_levels_>::solve(const real or_term) noexcept {
  real max_delta = poisson_pgs_or(or_term);
  if constexpr (mg_levels_ > 1) {
    residual();
    multilev_.restrict(delta_);
    multilev_.solve(or_term);
    // This isn't strictly the maximum delta, but it does provide an
    // upper bound on it which goes to zero as the solution converges
    max_delta += multilev_.prolongate(*this);
  }
  return max_delta;
}

template <int mg_levels_>
void PoissonFVMGSolver<mg_levels_>::restrict(const Mesh &src) noexcept {
  for (int i = 0; i < this->x_cells(); i++) {
    for (int j = 0; j < this->y_cells(); j++) {
      this->cv_average(i, j) = 0.25 * (src.cv_average(2 * i, 2 * j) +
                                       src.cv_average(2 * i, 2 * j + 1) +
                                       src.cv_average(2 * i + 1, 2 * j) +
                                       src.cv_average(2 * i + 1, 2 * j + 1));
    }
  }
}

template <int mg_levels_>
real PoissonFVMGSolver<mg_levels_>::prolongate(Mesh &dest) const noexcept {
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
