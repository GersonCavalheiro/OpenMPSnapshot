#pragma once

#include <nlohmann/detail/iterators/primitive_iterator.hpp>

namespace nlohmann
{
namespace detail
{

template<typename BasicJsonType> struct internal_iterator
{
typename BasicJsonType::object_t::iterator object_iterator {};
typename BasicJsonType::array_t::iterator array_iterator {};
typename BasicJsonType::binary_t::container_type::iterator binary_iterator {};
primitive_iterator_t primitive_iterator {};
};
}  
}  
