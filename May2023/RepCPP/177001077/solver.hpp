
#pragma once

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "defs.hpp"

#include "data.hpp"
#include "grid.hpp"

#if defined(NDEBUG) and not defined(MARLIN_PRINT_DEBUG_MSGS)
#    define MARLIN_DEBUG(x) ;
#else
#    include <iostream>
#    define MARLIN_DEBUG(x) x
#endif

namespace marlin
{
using grid_t = grid::grid_t<dim, index_t, scalar_t>;
using data_t = data::data_t<index_t, scalar_t>;

namespace solver
{
enum class boundary_condition_t
{
extrapolate,

periodic,

none,
};

struct params_t
{
scalar_t maxval;
scalar_t tolerance;
};

namespace detail
{
template<index_t N, typename T>
constexpr std::array<T, N> make_array(T init)
{
std::array<T, N> a;

for(index_t i = 0; i < N; ++i)
{
a[i] = init;
}

return a;
}
}    

class solver_t
{
public:
solver_t(
std::vector<scalar_t>&& rhs,
point_t const& dimensions,
std::array<std::pair<scalar_t, scalar_t>, dim> const& vertices,
params_t const& params,
std::array<boundary_condition_t, n_boundaries> const&
boundary_conditions = detail::make_array<n_boundaries>(
boundary_condition_t::extrapolate));

solver_t(solver_t&&) noexcept = default;
solver_t& operator=(solver_t&&) noexcept = default;
~solver_t() = default;

template<typename Hamiltonian, typename Viscosity>
void solve(Hamiltonian const& hamiltonian,
Viscosity const& viscosity) noexcept;

std::vector<scalar_t>&& steal() noexcept;

private:
void initialize() noexcept;

void compute_levels() noexcept;

void compute_bdry_idxs();

template<typename Hamiltonian, typename Viscosity>
bool iterate(Hamiltonian const& hamiltonian,
Viscosity const& viscosity) noexcept;

template<typename Hamiltonian, typename Viscosity>
scalar_t sweep(int dir,
Hamiltonian const& hamiltonian,
Viscosity const& viscosity) noexcept;

template<typename Hamiltonian, typename Viscosity>
scalar_t boundary(Hamiltonian const& hamiltonian,
Viscosity const& viscosity) noexcept;

scalar_t boundary_sweep_extrapolate(index_t boundary) noexcept;

template<typename Hamiltonian, typename Viscosity>
scalar_t boundary_sweep_periodic(
index_t boundary,
Hamiltonian const& hamiltonian,
Viscosity const& viscosity) noexcept;

template<typename Hamiltonian,
typename Viscosity,
typename GradientEstimator>
scalar_t update_point(int dir,
point_t point,
Hamiltonian const& hamiltonian,
Viscosity const& viscosity,
GradientEstimator const& ge) noexcept;

grid_t m_grid;    
data_t m_soln;    
data_t m_cost;    

scalar_t m_tolerance;

std::vector<std::vector<point_t>> m_levels;

std::array<std::vector<index_t>, n_boundaries> m_bdry_idxs;

std::array<boundary_condition_t, n_boundaries> m_bc;
};

namespace detail
{
struct update_data_internal_t
{
vector_t p;
vector_t avgs;
};

inline update_data_internal_t estimate_p_interior(
point_t const& point,
data_t const& soln,
grid_t const& grid) noexcept
{
update_data_internal_t res;

for(auto i = 0; i < dim; ++i)
{
auto neighbor = point;

neighbor[i] += 1;
auto const right = soln.at(grid.index(neighbor));

neighbor[i] -= 2;
auto const left = soln.at(grid.index(neighbor));

res.p[i] = (right - left) / (scalar_t{ 2.0 } * grid.h(i));
res.avgs[i] =
(right + left) / (scalar_t{ 2.0 } * grid.h(i));
}

return res;
}

inline update_data_internal_t estimate_p_boundary(
point_t const& point,
data_t const& soln,
grid_t const& grid) noexcept
{
update_data_internal_t res;

for(auto i = 0; i < dim; ++i)
{
auto neighbor = point;
auto const original_idx = point[i];

neighbor[i] += 1;

auto const right = soln.at(grid.index(neighbor));

if(original_idx > 0)
neighbor[i] = original_idx - 1;
else
neighbor[i] = grid.size(i) - 2;

auto const left = soln.at(grid.index(neighbor));

res.p[i] = (right - left) / (scalar_t{ 2.0 } * grid.h(i));
res.avgs[i] =
(right + left) / (scalar_t{ 2.0 } * grid.h(i));
}

return res;
}

inline scalar_t update(scalar_t ham_value,
scalar_t scale,
scalar_t cost,
vector_t const& avgs,
vector_t const& viscosity) noexcept
{
assert(cost > 0);
return scale * (cost - ham_value +
std::inner_product(std::begin(viscosity),
std::end(viscosity),
std::begin(avgs),
0.0));
}

inline scalar_t scale(vector_t const& viscosity,
vector_t const& h) noexcept
{
return scalar_t{ 1.0 } /
std::inner_product(std::begin(viscosity),
std::end(viscosity),
std::begin(h),
0.0,
std::plus<>(),
std::divides<>());
}
}    

template<typename Hamiltonian, typename Viscosity>
void solver_t::solve(Hamiltonian const& hamiltonian,
Viscosity const& viscosity) noexcept
{
MARLIN_DEBUG(auto niter = 0;
std::cerr << "Iteration " << niter++ << ":\n";)

while(!iterate(hamiltonian, viscosity))
{
MARLIN_DEBUG(std::cerr << "Iteration " << niter++ << ":\n";)
}
}

template<typename Hamiltonian, typename Viscosity>
bool solver_t::iterate(Hamiltonian const& hamiltonian,
Viscosity const& viscosity) noexcept
{
for(auto dir = 0; dir < n_sweeps; ++dir)
{
scalar_t diff = 0;

diff = sweep(dir, hamiltonian, viscosity);
MARLIN_DEBUG(std::cerr << "Sweep " << dir
<< ": delta = " << diff << '\n';)
diff = std::max(diff, boundary(hamiltonian, viscosity));
MARLIN_DEBUG(std::cerr << "Sweep " << dir
<< " (after boundary): delta = " << diff
<< '\n';)
if(diff < m_tolerance)
{
return true;
}
}

return false;
}

template<typename Hamiltonian, typename Viscosity>
scalar_t solver_t::sweep(int dir,
Hamiltonian const& hamiltonian,
Viscosity const& viscosity) noexcept
{
assert(dir >= 0);
assert(dir < n_sweeps);

scalar_t diff = 0;

for(auto const& level : m_levels)
{
std::size_t const size = level.size();
std::vector<scalar_t> delta(size);

#pragma omp parallel default(none) \
shared(delta, level, dir, hamiltonian, viscosity, diff, size)
{
#pragma omp for schedule(static) nowait
for(auto i = 0ul; i < size; ++i)
{
delta[i] = update_point(dir,
level[i],
hamiltonian,
viscosity,
detail::estimate_p_interior);
}

#pragma omp for schedule(static) reduction(max : diff) nowait
for(std::size_t i = 0; i < size; ++i)
diff = std::max(delta[i], diff);
}
}

return diff;
}

template<typename Hamiltonian,
typename Viscosity,
typename GradientEstimator>
scalar_t solver_t::update_point(int dir,
point_t point,
Hamiltonian const& hamiltonian,
Viscosity const& viscosity,
GradientEstimator const& ge) noexcept
{
point = m_grid.rotate_axes(point, dir);
index_t const index = m_grid.index(point);
scalar_t const cost = m_cost.at(index);

if(cost < scalar_t{ 0.0 })
{
return scalar_t{ 0.0 };
}

auto const data = ge(point, m_soln, m_grid);

scalar_t const old = m_soln.at(index);

#ifdef MARLIN_USE_ROWMAJOR
vector_t const sigma = viscosity(index);
scalar_t const hval = hamiltonian(index, data.p);
#else
vector_t const sigma = viscosity(point);
scalar_t const hval = hamiltonian(point, data.p);
#endif
scalar_t const scale_ = detail::scale(sigma, m_grid.h());

scalar_t const new_val = m_soln.at(index) = std::min(
detail::update(hval, scale_, cost, data.avgs, sigma), old);

return old - new_val;
}

template<typename Hamiltonian, typename Viscosity>
scalar_t solver_t::boundary(Hamiltonian const& hamiltonian,
Viscosity const& viscosity) noexcept
{
scalar_t diff = 0;

for(index_t bdry = 0; bdry < n_boundaries; ++bdry)
{
switch(m_bc[bdry])
{
case boundary_condition_t::extrapolate:
diff = std::max(diff, boundary_sweep_extrapolate(bdry));
break;
case boundary_condition_t::periodic:
diff = std::max(diff,
boundary_sweep_periodic(
bdry, hamiltonian, viscosity));
break;
case boundary_condition_t::none:
break;
}
}

return diff;
}

template<typename Hamiltonian, typename Viscosity>
scalar_t solver_t::boundary_sweep_periodic(
index_t boundary,
Hamiltonian const& hamiltonian,
Viscosity const& viscosity) noexcept
{
assert(boundary < n_boundaries);

scalar_t delta = 0.0;

for(index_t index : m_bdry_idxs[boundary])
{
auto const point = m_grid.point(index);

if(boundary < dim)
{
delta = std::max(delta,
update_point(0,
point,
hamiltonian,
viscosity,
detail::estimate_p_boundary));
}
else
{
auto opposing_point = point;
opposing_point[boundary - dim] = 0;
m_soln.at(index) = m_soln.at(m_grid.index(opposing_point));

}
}

return delta;
}
}    
}    
