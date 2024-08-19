

#pragma once

#include <array>
#include <stddef.h>


template<typename T, class InputIt, class OutputIt, size_t sizeFactors, typename... Targs>
constexpr void cartesian_product_impl( OutputIt& product,
size_t indexFactor,
std::array<T, sizeFactors>& currentElement,
InputIt beginInput, InputIt endInput )
{
for( ; beginInput != endInput; ++beginInput )
{
currentElement[ indexFactor ] = *beginInput;
*product++ = currentElement;
}
}

template<typename T, class InputIt, class OutputIt, size_t sizeFactors, typename... Targs>
constexpr void cartesian_product_impl( OutputIt& product,
size_t indexFactor,
std::array<T, sizeFactors>& currentElement,
InputIt beginInput, InputIt endInput, 
Targs... args )
{
for( ; beginInput != endInput; ++beginInput )
{
currentElement[ indexFactor ] = *beginInput;
cartesian_product_impl( product, indexFactor + 1, currentElement, args... );
}
}





template<class InputIt, class OutputIt, typename... Targs>
constexpr void cartesian_product( OutputIt product,
InputIt beginInput, InputIt endInput, 
Targs... args )
{
using T = typename std::iterator_traits<InputIt>::value_type;
std::array<T, ( sizeof...( Targs ) / 2 ) + 1 > currentElement;
size_t indexFactor = 0;

for( ; beginInput != endInput; ++beginInput )
{
currentElement[ indexFactor ] = *beginInput;
cartesian_product_impl( product, indexFactor + 1, currentElement, args... );
}
}
