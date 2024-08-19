

#pragma once

#include <type_traits>

namespace alpaka::meta
{
template<typename T>
struct DependentFalseType : std::false_type
{
};
} 
