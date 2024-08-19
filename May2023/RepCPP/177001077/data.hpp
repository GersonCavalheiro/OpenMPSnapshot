
#pragma once

#include <cassert>
#include <vector>

namespace marlin
{
namespace data
{
template<typename Index, typename Scalar>
class data_t
{
public:
using index_t = Index;
using scalar_t = Scalar;

explicit data_t(std::vector<scalar_t>&& values) noexcept
: m_values(std::move(values)), m_memsize(m_values.size())
{}

data_t(index_t memsize, scalar_t fill = scalar_t{ 0 }) noexcept
: m_values(std::vector<scalar_t>(memsize)), m_memsize(memsize)
{
std::fill(std::begin(m_values), std::end(m_values), fill);
}

data_t(data_t const&) = delete;
data_t& operator=(data_t const&) = delete;

data_t(data_t&&) noexcept = default;
data_t& operator=(data_t&&) noexcept = default;

~data_t() = default;

scalar_t at(index_t index) const noexcept
{
assert(index < m_memsize);
return m_values[index];
}

scalar_t& at(index_t index) noexcept
{
assert(index < m_memsize);
return m_values[index];
}

scalar_t* get_values() const noexcept { return m_values.data(); }

std::vector<scalar_t>&& steal() noexcept
{
return std::move(m_values);
}

index_t size() const noexcept { return m_memsize; }

private:
std::vector<scalar_t> m_values;
index_t m_memsize;
};
}    
}    
