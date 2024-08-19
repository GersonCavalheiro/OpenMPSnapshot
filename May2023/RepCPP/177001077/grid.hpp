
#pragma once

#include <array>
#include <cassert>
#include <numeric>

namespace marlin
{
namespace grid
{
template<typename Index>
inline bool backwards(Index dir, unsigned dimension) noexcept
{
assert(dir < n_sweeps);
assert(dimension < dim);

return static_cast<bool>(dir & (1 << dimension));
}

template<unsigned dim, typename Index>
inline Index rotate(Index boundary_dim, unsigned dimension) noexcept
{
assert(boundary_dim < dim);
assert(dimension < dim - 1);

auto idx = boundary_dim + dimension + 1;

return idx >= dim ? idx - dim : idx;
}

template<int dim, typename Index, typename Scalar>
struct grid_t
{
public:
using index_t = Index;
using point_t = std::array<index_t, dim>;
using scalar_t = Scalar;
using vector_t = std::array<scalar_t, dim>;

static_assert(dim >= 2,
"Number of dimensions must be at least two.");

grid_t(
std::array<std::pair<scalar_t, scalar_t>, dim> const& vertices,
point_t const& size) noexcept
: m_size(size),
m_npts(std::accumulate(std::begin(m_size),
std::end(m_size),
1,
std::multiplies<>())),
m_h([&] {
vector_t h;

for(auto i = 0; i < dim; ++i)
{
h[i] = (vertices[i].second - vertices[i].first) /
(m_size[i] - 1);
}

return h;
}()),
m_nlevels(std::accumulate(std::begin(m_size),
std::end(m_size),
0ul) -
dim + 1)
{
assert([this] {
for(auto i = 0; i < dim; ++i)
{
if(m_h[i] <= 0)
return false;
}
return true;
}());
}

grid_t(grid_t const&) noexcept = default;
grid_t& operator=(grid_t const&) noexcept = default;

grid_t(grid_t&&) noexcept = default;
grid_t& operator=(grid_t&&) noexcept = default;

~grid_t() = default;

index_t size(index_t i) const noexcept
{
assert(i < dim);
return m_size[i];
}

point_t const& size() const noexcept { return m_size; }
index_t npts() const noexcept { return m_npts; }

scalar_t h(index_t i) const noexcept
{
assert(i < dim);
return m_h[i];
}
vector_t const& h() const noexcept { return m_h; }

index_t n_levels() const noexcept { return m_nlevels; }

point_t point(index_t index) const noexcept
{
assert(index < m_npts);

point_t point;

for(auto i = dim; i > 0; --i)
{
point[i - 1] = index % m_size[i - 1];
index /= m_size[i - 1];
}

return point;
}

index_t index(point_t const& point) const noexcept
{
assert([&] {
for(auto i = 0; i < dim; ++i)
if(point[i] >= m_size[i])
return false;
return true;
}());

index_t offset = 0;

for(auto i = 0; i < dim - 1; ++i)
{
offset = m_size[i + 1] * (point[i] + offset);
}

offset += point[dim - 1];

return offset;
}

point_t rotate_axes(point_t point, int dir) const noexcept
{
for(int i = 0; i < dim; ++i)
{
if(backwards(dir, i))
{
point[i] = m_size[i] - point[i] - 1;
}
}

return point;
}

bool is_boundary(point_t const& point) const noexcept
{
for(auto i = 0; i < dim; ++i)
{
if(point[i] == 0 || point[i] == m_size[i] - 1)
{
return true;
}
}

return false;
}

bool is_boundary_interior(index_t bdry,
point_t const& point) const noexcept
{
int const bdry_dim = bdry % dim;

for(int i = 0; i < dim; ++i)
{
if(i != bdry_dim &&
(point[i] == 0 || point[i] == m_size[i] - 1))
return false;
}

return true;
}

index_t next(index_t idx, index_t dir) const noexcept
{
assert(idx <= m_npts);
assert(dir < n_sweeps);

if(idx == m_npts)
{
point_t p;

for(auto i = 0; i < dim; ++i)
{
p[i] = backwards(dir, i) ? m_size[i] - 2 : 1;
}

idx = index(p);
}
else
{
point_t p = point(idx);

p[0] += backwards(dir, 0) ? -1 : +1;

for(auto i = 0; i < dim - 1; ++i)
{
if(p[i] != 0 && p[i] != m_size[i] - 1)
{
break;
}

p[i] = backwards(dir, i) ? m_size[i] - 2 : 1;
p[i + 1] += backwards(dir, i + 1) ? -1 : +1;
}

if(p[dim - 1] == 0 || p[dim - 1] == m_size[dim - 1] - 1)
{
idx = m_npts;
}
else
{
idx = index(p);
}
}

return idx;
}

index_t next_in_boundary(index_t idx,
index_t boundary) const noexcept
{
assert(idx <= m_npts);
assert(boundary < dim);

if(idx == m_npts)
return index_t{ 0 };

point_t p = point(idx);

p[rotate<dim>(boundary, 0)] += 1;

for(auto i = 0; i < dim - 2; ++i)
{
if(p[rotate<dim>(boundary, i)] ==
m_size[rotate<dim>(boundary, i)])
{
p[rotate<dim>(boundary, i)] = 0;
p[rotate<dim>(boundary, i + 1)] += 1;
}
}

if(p[rotate<dim>(boundary, dim - 2)] ==
m_size[rotate<dim>(boundary, dim - 2)])
return m_npts;

else
return index(p);
}

private:
point_t m_size;
index_t m_npts;
vector_t m_h;
index_t m_nlevels;
};
}    
}    
