#pragma once

#include <memory>
#include <vector>

#include <allocator_cu.hpp>

namespace backends
{
template<typename T, typename Alloc = details::allocator_cu<T> >
using vector = std::vector<T, Alloc>;
}