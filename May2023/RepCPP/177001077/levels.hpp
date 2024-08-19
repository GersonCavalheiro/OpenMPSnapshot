
#pragma once

#include <cassert>
#include "marlin/defs.hpp"

#ifndef NDEBUG
#    include <numeric>
#endif

namespace marlin
{
namespace level
{
template<index_t N>
class level_t
{
static_assert(N <= marlin::dim, "");
friend class level_t<N + 1>;

public:
level_t(index_t sum, point_t const& limits) noexcept;

void get(index_t* range) const noexcept;

bool next() noexcept;

private:
bool reset(index_t sum) noexcept;
index_t sum() noexcept;

index_t m_limit;
index_t m_value;
level_t<N - 1> m_sublevel;
};

template<>
class level_t<1>
{
friend class level_t<2>;

public:
level_t(index_t sum, point_t const& limits) noexcept;

void get(index_t* range) const noexcept;
bool next() const noexcept;

private:
bool reset(index_t sum) noexcept;
index_t sum() noexcept;

index_t m_value;
index_t m_limit;
};

template<index_t N>
level_t<N>::level_t(index_t sum, point_t const& limits) noexcept
: m_limit(limits[dim - N]),
m_value(std::min(sum, m_limit - 1)),
m_sublevel(sum > m_value ? sum - m_value : 0, limits)
{
assert(std::accumulate(
std::begin(limits) + dim - N, std::end(limits), 0) -
N >=
sum);
}

level_t<1>::level_t(index_t sum, point_t const& limits) noexcept
: m_value(sum), m_limit(limits.back())
{
assert(m_value < m_limit);
}

template<index_t N>
void level_t<N>::get(index_t* range) const noexcept
{
assert(range != nullptr);

*range = m_value;
m_sublevel.get(range + 1);
}

void level_t<1>::get(index_t* range) const noexcept
{
assert(range != nullptr);
*range = m_value;
}

template<index_t N>
bool level_t<N>::next() noexcept
{
if(m_sublevel.next())
return true;

if(m_value == 0)
return false;

m_value--;
return m_sublevel.reset(m_sublevel.sum() + 1);
}

bool level_t<1>::next() const noexcept { return false; }

template<index_t N>
bool level_t<N>::reset(index_t sum) noexcept
{
m_value = std::min(sum, m_limit - 1);
return m_sublevel.reset(sum > m_value ? sum - m_value : 0);
}

bool level_t<1>::reset(index_t sum) noexcept
{
m_value = sum;
return m_value < m_limit;
}

template<index_t N>
index_t level_t<N>::sum() noexcept
{
return m_value + m_sublevel.sum();
}

index_t level_t<1>::sum() noexcept { return m_value; }

}    
}    
