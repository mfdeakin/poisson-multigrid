
#ifndef _POISSON_HPP_
#define _POISSON_HPP_

#include <functional>

#include "xtensor/xtensor.hpp"

using real = double;

constexpr real pi = 3.1415926535897932;

class Mesh {
public:
  // Assume a single ghost cell on each side of the boundary
  static constexpr int ghost_cells = 1;

  // Cells are (externally) zero referenced from the bottom left corners
  // interior cell
  constexpr real x1(int cell_x) const noexcept { return min_x_ + cell_x * dx_; }

  constexpr real x2(int cell_x) const noexcept {
    return min_x_ + (cell_x + 1) * dx_;
  }

  constexpr real y1(int cell_y) const noexcept { return min_y_ + cell_y * dy_; }

  constexpr real y2(int cell_y) const noexcept {
    return min_y_ + (cell_y + 1) * dy_;
  }

  constexpr real median_x(int cell_x) const noexcept {
    return min_x_ + dx_ / 2 + cell_x * dx_;
  }

  constexpr real median_y(int cell_y) const noexcept {
    return min_y_ + dy_ / 2 + cell_y * dy_;
  }

  constexpr int x_idx(real x) const noexcept {
    return static_cast<int>(x / dx_ - min_x_ / dx_);
  }

  constexpr int y_idx(real y) const noexcept {
    return static_cast<int>(y / dy_ - min_y_ / dx_);
  }

  Mesh(const std::pair<real, real> &corner_1,
       const std::pair<real, real> &corner_2, const size_t x_cells,
       const size_t y_cells) noexcept;

  Mesh(const std::pair<real, real> &corner_1,
       const std::pair<real, real> &corner_2, const size_t x_cells,
       const size_t y_cells, std::function<real(real, real)> f) noexcept;

  real cv_average(int i, int j) const noexcept {
    return cva_(i + ghost_cells, j + ghost_cells);
  }
  real &cv_average(int i, int j) noexcept {
    return cva_(i + ghost_cells, j + ghost_cells);
  }

  real operator[](std::pair<int, int> idx) const noexcept {
    return cv_average(idx.first, idx.second);
  }
  real &operator[](std::pair<int, int> idx) noexcept {
    return cv_average(idx.first, idx.second);
  }

  real interpolate(real x, real y) const noexcept;
  real operator()(real x, real y) const noexcept { return interpolate(x, y); }

  int x_cells() const noexcept { return cva_.shape()[0] - 2 * ghost_cells; }
  int y_cells() const noexcept { return cva_.shape()[1] - 2 * ghost_cells; }

protected:
  real min_x_, max_x_, min_y_, max_y_;
  real dx_, dy_;

  xt::xtensor<real, 2> cva_;
};

class BoundaryConditions {
public:
  BoundaryConditions(const BoundaryConditions &src,
                     const std::function<real(real, real)> &bc) noexcept
      : dirichlet_(src.dirichlet_), neumann_(src.neumann_), bc_(bc) {}

  void apply(Mesh &mesh) const noexcept;

protected:
  // This class is to be subclassed and constructed there,
  // adding cell indices on the boundaries which are dirichlet
  // and neumann to their respective vectors
  explicit BoundaryConditions(
      const std::function<real(real, real)> &bc) noexcept
      : dirichlet_(), neumann_(), bc_(bc) {}

  std::pair<int, int> adjacent(const std::pair<int, int> &ghost_idx, int max_x,
                               int max_y) const noexcept;

  // A list of indices for cells in the boundary with the specified boundary
  // condition
  std::vector<std::pair<int, int>> dirichlet_;
  std::vector<std::pair<int, int>> neumann_;
  std::function<real(real, real)> bc_;
};

template <int mg_levels_> class PoissonFVMGSolver : public Mesh {
public:
  using Coarsen = PoissonFVMGSolver<mg_levels_ - 1>;

  PoissonFVMGSolver(const std::pair<real, real> &corner_1,
                    const std::pair<real, real> &corner_2, const size_t x_cells,
                    const size_t y_cells,
                    const std::function<real(real, real)> &source) noexcept;

  real solve(const real or_term = 1.5) noexcept;
  void restrict(const Mesh &src) noexcept;
  real prolongate(Mesh &dest) const noexcept;

  template <int> friend class PoissonFVMGSolver;

protected:
  real delta(const int i, const int j) const noexcept;
  void residual() noexcept;
  real poisson_pgs_or(const real or_term = 1.5) noexcept;

  Coarsen multilev_;

  BoundaryConditions bc_;

  Mesh source_;
  Mesh delta_;
};

template <> class PoissonFVMGSolver<1> : public Mesh {
  PoissonFVMGSolver(const std::pair<real, real> &corner_1,
                    const std::pair<real, real> &corner_2, const size_t x_cells,
                    const size_t y_cells,
                    const std::function<real(real, real)> &source) noexcept;

  real solve(const real or_term = 1.5) noexcept;
  void restrict(const Mesh &src) noexcept;
  real prolongate(Mesh &dest) const noexcept;

  template <int> friend class PoissonFVMGSolver;

protected:
  real delta(const int i, const int j) const noexcept;
  void residual() noexcept;
  real poisson_pgs_or(const real or_term = 1.5) noexcept;

  BoundaryConditions bc_;

  Mesh source_;
  Mesh delta_;
};

#endif
