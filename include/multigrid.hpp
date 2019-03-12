
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
  constexpr real left_x(int cell_x) const noexcept {
    return min_x_ + cell_x * dx_;
  }

  constexpr real right_x(int cell_x) const noexcept {
    return min_x_ + (cell_x + 1) * dx_;
  }

  constexpr real bottom_y(int cell_y) const noexcept {
    return min_y_ + cell_y * dy_;
  }

  constexpr real top_y(int cell_y) const noexcept {
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
    return static_cast<int>(y / dy_ - min_y_ / dy_);
  }

  constexpr real dx() const noexcept { return dx_; }

  constexpr real dy() const noexcept { return dy_; }

  Mesh(const std::pair<real, real> &corner_1,
       const std::pair<real, real> &corner_2, const size_t x_cells,
       const size_t cells_y) noexcept;

  Mesh(const std::pair<real, real> &corner_1,
       const std::pair<real, real> &corner_2, const size_t x_cells,
       const size_t cells_y, std::function<real(real, real)> f) noexcept;

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

  int cells_x() const noexcept { return cva_.shape()[0] - 2 * ghost_cells; }
  int cells_y() const noexcept { return cva_.shape()[1] - 2 * ghost_cells; }

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
      : left_t_(BC_Type::dirichlet),
        right_t_(BC_Type::dirichlet),
        top_t_(BC_Type::dirichlet),
        bottom_t_(BC_Type::dirichlet),
        left_bc_(&homogeneous),
        right_bc_(&homogeneous),
        top_bc_(&homogeneous),
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

class PoissonFVMGSolverBase : public Mesh {
 public:
  PoissonFVMGSolverBase(const std::pair<real, real> &corner_1,
                        const std::pair<real, real> &corner_2,
                        const size_t cells_x, const size_t cells_y,
                        const BoundaryConditions &bc) noexcept;

  PoissonFVMGSolverBase(const std::pair<real, real> &corner_1,
                        const std::pair<real, real> &corner_2,
                        const size_t cells_x, const size_t cells_y,
                        const BoundaryConditions &bc,
                        const std::function<real(real, real)> &source) noexcept;

  void restrict(const PoissonFVMGSolverBase &src) noexcept;
  real prolongate(Mesh &dest) const noexcept;

  const BoundaryConditions &bconds() const noexcept { return bc_; }
  const Mesh &source() const noexcept { return source_; }
  real poisson_pgs_or(const real or_term = 1.5) noexcept;
  real delta(const int i, const int j) const noexcept;

 protected:
  BoundaryConditions bc_;
  Mesh source_;
};

template <int mg_levels_>
class PoissonFVMGSolver : public PoissonFVMGSolverBase {
 public:
  PoissonFVMGSolver(const std::pair<real, real> &corner_1,
                    const std::pair<real, real> &corner_2, const size_t cells_x,
                    const size_t cells_y,
                    const BoundaryConditions &bc) noexcept;

  PoissonFVMGSolver(const std::pair<real, real> &corner_1,
                    const std::pair<real, real> &corner_2, const size_t cells_x,
                    const size_t cells_y, const BoundaryConditions &bc,
                    const std::function<real(real, real)> &source) noexcept;

  using Coarsen = PoissonFVMGSolver<mg_levels_ - 1>;

  template <typename Iterable>
  real solve(const Iterable &iterations) noexcept {
    // Iterable must be a container of triples, specifying in order the level to
    // run pgs at, the number of iterations to run at that level, and the
    // over-relaxation parameter to use
    auto [end, max_delta] = solve_int(iterations.begin(), iterations.end());
    assert(end == iterations.end());
    return max_delta;
  }

  const Coarsen &error_mesh() { return multilev_; };

  template <int>
  friend class PoissonFVMGSolver;

 protected:
  template <typename Iter>
  typename std::enable_if<
      std::is_same<typename std::iterator_traits<Iter>::value_type,
                   std::tuple<int, int, real>>::value,
      std::pair<Iter, real>>::type
  solve_int(Iter iterations, const Iter &end) noexcept {
    // iterations is an iterator over a container of triples (tuples)
    // The tuple contains the level to smooth at, the number of smoothing
    // passes, and the over-relaxation parameter to smooth with
    real max_delta = 0.0;
    while(iterations != end && std::get<0>(*iterations) <= mg_levels_) {
      const int level = std::get<0>(*iterations);
      if(level == mg_levels_) {
        const int num_iter = std::get<1>(*iterations);
        const real or_term = std::get<2>(*iterations);
        for(int i = 0; i < num_iter; i++) {
          max_delta += poisson_pgs_or(or_term);
        }
        iterations++;
      } else {
        // level < mg_levels_; run pgs at a more coarse scale
        // We need the boundary conditions to hold when computing the residual
        bc_.apply(*this);
        multilev_.restrict(*this);
        auto [iter_processed, delta] = multilev_.solve_int(iterations, end);
        multilev_.bc_.apply(multilev_);
        max_delta += multilev_.prolongate(*this);
        iterations = iter_processed;
      }
    }
    // This isn't strictly the maximum delta, but it does provide an
    // upper bound on it which goes to zero as the solution converges
    return {iterations, max_delta};
  }

  Coarsen multilev_;
};

template <>
class PoissonFVMGSolver<1> : public PoissonFVMGSolverBase {
 public:
  static constexpr int mg_levels_ = 1;

  PoissonFVMGSolver(const std::pair<real, real> &corner_1,
                    const std::pair<real, real> &corner_2, const size_t cells_x,
                    const size_t cells_y,
                    const BoundaryConditions &bc) noexcept;

  PoissonFVMGSolver(const std::pair<real, real> &corner_1,
                    const std::pair<real, real> &corner_2, const size_t cells_x,
                    const size_t cells_y, const BoundaryConditions &bc,
                    const std::function<real(real, real)> &source) noexcept;

  // Each pair specifies the level the operation is to be performed on as the
  // first parameter, and the number of pgs-or iterations to be performed on
  // that level as the second parameter
  template <typename Iterable>
  real solve(const Iterable &iterations) noexcept {
    auto [end, max_delta] = solve_int(iterations.begin(), iterations.end());
    assert(end == iterations.end());
    return max_delta;
  }

  template <int>
  friend class PoissonFVMGSolver;

 protected:
  template <typename Iter>
  typename std::enable_if<
      std::is_same<typename std::iterator_traits<Iter>::value_type,
                   std::tuple<int, int, real>>::value,
      std::pair<Iter, real>>::type
  solve_int(Iter iterations, const Iter &end) noexcept {
    // iterations is an iterator over a container of triples (tuples)
    // The tuple contains the level to smooth at, the number of smoothing
    // passes, and the over-relaxation parameter to smooth with
    real max_delta = 0.0;
    while(iterations != end && std::get<0>(*iterations) <= 1) {
      assert(std::get<0>(*iterations) == mg_levels_);
      const int num_iter = std::get<1>(*iterations);
      const real or_term = std::get<2>(*iterations);
      for(int i = 0; i < num_iter; i++) {
        max_delta += poisson_pgs_or(or_term);
      }
      iterations++;
    }
    // This isn't strictly the maximum delta, but it does provide an
    // upper bound on it which goes to zero as the solution converges
    return {iterations, max_delta};
  }
};

#endif
