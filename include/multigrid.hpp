
#ifndef _POISSON_HPP_
#define _POISSON_HPP_

#include <functional>
#include <iterator>

#include "xtensor/xtensor.hpp"

using real = double;

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

  const xt::xtensor<real, 2> data() const noexcept { return cva_; }

  int x_cells() const noexcept { return cva_.shape()[0] - 2 * ghost_cells; }
  int y_cells() const noexcept { return cva_.shape()[1] - 2 * ghost_cells; }

protected:
  real min_x_, max_x_, min_y_, max_y_;
  real dx_, dy_;

  xt::xtensor<real, 2> cva_;
};

class BoundaryConditions {
public:
  enum class BC_Type { dirichlet, neumann };

  static real homogeneous(real, real) { return 0.0; }

  BoundaryConditions()
      : left_t_(BC_Type::dirichlet), right_t_(BC_Type::dirichlet),
        top_t_(BC_Type::dirichlet), bottom_t_(BC_Type::dirichlet),
        left_bc_(&homogeneous), right_bc_(&homogeneous), top_bc_(&homogeneous),
        bottom_bc_(&homogeneous) {}
  BoundaryConditions(
      const BC_Type left_t, const std::function<real(real, real)> &left_bc,
      const BC_Type right_t, const std::function<real(real, real)> &right_bc,
      const BC_Type top_t, const std::function<real(real, real)> &top_bc,
      const BC_Type bottom_t, const std::function<real(real, real)> &bottom_bc);

  void apply(Mesh &mesh) const noexcept;

  std::pair<BC_Type, std::function<real(real, real)>> left_bc() const noexcept {
    return {left_t_, left_bc_};
  }
  std::pair<BC_Type, std::function<real(real, real)>> right_bc() const
      noexcept {
    return {right_t_, right_bc_};
  }
  std::pair<BC_Type, std::function<real(real, real)>> top_bc() const noexcept {
    return {top_t_, top_bc_};
  }
  std::pair<BC_Type, std::function<real(real, real)>> bottom_bc() const
      noexcept {
    return {bottom_t_, bottom_bc_};
  }

protected:
  BC_Type left_t_, right_t_, top_t_, bottom_t_;
  std::function<real(real, real)> left_bc_;
  std::function<real(real, real)> right_bc_;
  std::function<real(real, real)> top_bc_;
  std::function<real(real, real)> bottom_bc_;
};

template <int mg_levels_> class PoissonFVMGSolver : public Mesh {
public:
  using Coarsen = PoissonFVMGSolver<mg_levels_ - 1>;

  PoissonFVMGSolver(const std::pair<real, real> &corner_1,
                    const std::pair<real, real> &corner_2, const size_t x_cells,
                    const size_t y_cells, BoundaryConditions &bc,
                    const std::function<real(real, real)> &source) noexcept;

  template <typename Iterable>
  real solve(const Iterable &iterations, const real or_term = 1.5) noexcept {
    auto [end, max_delta] =
        solve_int(iterations.begin(), iterations.end(), or_term);
    assert(end == iterations.end());
    return max_delta;
  }

  void restrict(const Mesh &src) noexcept;
  real prolongate(Mesh &dest) const noexcept;

  const Mesh &source() const noexcept { return source_; }

  template <int> friend class PoissonFVMGSolver;

protected:
  template <typename Iter>
  typename std::enable_if<
      std::is_same<typename std::iterator_traits<Iter>::value_type,
                   std::pair<int, int>>::value,
      std::pair<Iter, real>>::type
  solve_int(Iter iterations, const Iter &end, const real or_term) noexcept {
    auto [level, num_iter] = *iterations;
    real max_delta = 0.0;
    if (level == mg_levels_) {
      for (int i = 0; i < num_iter; i++) {
        max_delta += poisson_pgs_or(or_term);
      }
      iterations++;
      if (iterations == end) {
        return {iterations, max_delta};
      }
      level = iterations->first;
      if (level > mg_levels_) {
        return {iterations, max_delta};
      }
    }
    residual();
    multilev_.restrict(delta_);
    multilev_.solve_int(iterations, end, or_term);
    // This isn't strictly the maximum delta, but it does provide an
    // upper bound on it which goes to zero as the solution converges
    max_delta += multilev_.prolongate(*this);
    return {iterations, max_delta};
  }

  real delta(const int i, const int j) const noexcept;
  void residual() noexcept;
  real poisson_pgs_or(const real or_term = 1.5) noexcept;

  Coarsen multilev_;

  BoundaryConditions bc_;

  Mesh source_;
  Mesh delta_;
};

template <> class PoissonFVMGSolver<1> : public Mesh {
public:
  PoissonFVMGSolver(const std::pair<real, real> &corner_1,
                    const std::pair<real, real> &corner_2, const size_t x_cells,
                    const size_t y_cells, BoundaryConditions &bc,
                    const std::function<real(real, real)> &source) noexcept;

  // Each pair specifies the level the operation is to be performed on as the
  // first parameter, and the number of pgs-or iterations to be performed on
  // that level as the second parameter
  template <typename Iterable>
  real solve(const Iterable &iterations, const real or_term = 1.5) noexcept {
    auto [end, max_delta] =
        solve_int(iterations.begin(), iterations.end(), or_term);
    assert(end == iterations.end());
    return max_delta;
  }

  void restrict(const Mesh &src) noexcept;
  real prolongate(Mesh &dest) const noexcept;

  const Mesh &source() const noexcept { return source_; }

  template <int> friend class PoissonFVMGSolver;

protected:
  template <typename Iter>
  typename std::enable_if<
      std::is_same<typename std::iterator_traits<Iter>::value_type,
                   std::pair<int, int>>::value,
      std::pair<Iter, real>>::type
  solve_int(Iter iterations, const Iter &end, const real or_term) noexcept {
    auto [level, num_iter] = *iterations;
    real max_delta = 0.0;
    assert(level == 1);
    for (int i = 0; i < num_iter; i++) {
      max_delta += poisson_pgs_or(or_term);
    }
    iterations++;
    if (iterations == end) {
      return {iterations, max_delta};
    }
    level = iterations->first;
    if (level > 1) {
      return {iterations, max_delta};
    }

    // Error!
    return {iterations, std::numeric_limits<real>::quiet_NaN()};
  }

  real delta(const int i, const int j) const noexcept;
  void residual() noexcept;
  real poisson_pgs_or(const real or_term = 1.5) noexcept;

  BoundaryConditions bc_;

  Mesh source_;
  Mesh delta_;
};

#endif
