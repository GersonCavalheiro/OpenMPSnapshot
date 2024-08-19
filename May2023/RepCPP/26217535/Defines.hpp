

#pragma once

#include <alpaka/alpaka.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

namespace alpaka
{
namespace test
{
namespace unit
{
namespace math
{
enum class Range
{
OneNeighbourhood,
PositiveOnly,
PositiveAndZero,
NotZero,
Unrestricted,
Anything
};

enum class Arity
{
Unary = 1,
Binary = 2
};

template<typename T, Arity Tarity>
struct ArgsItem
{
static constexpr Arity arity = Tarity;
static constexpr size_t arity_nr = static_cast<size_t>(Tarity);

T arg[arity_nr]; 

friend auto operator<<(std::ostream& os, ArgsItem const& argsItem) -> std::ostream&
{
os.precision(17);
os << "[ ";
for(size_t i = 0; i < argsItem.arity_nr; ++i)
os << std::setprecision(std::numeric_limits<T>::digits10 + 1) << argsItem.arg[i] << ", ";
os << "]";
return os;
}
};

template<typename T>
auto rsqrt(T const& arg)
{
using std::sqrt;
return static_cast<T>(1) / sqrt(arg);
}

template<typename TAcc, typename T>
ALPAKA_FN_HOST_ACC auto divides(TAcc&, T const& arg1, T const& arg2)
{
return arg1 / arg2;
}

template<typename TAcc, typename T>
ALPAKA_FN_HOST_ACC auto minus(TAcc&, T const& arg1, T const& arg2)
{
return arg1 - arg2;
}

template<typename TAcc, typename T>
ALPAKA_FN_HOST_ACC auto multiplies(TAcc&, T const& arg1, T const& arg2)
{
return arg1 * arg2;
}

template<typename TAcc, typename T>
ALPAKA_FN_HOST_ACC auto plus(TAcc&, T const& arg1, T const& arg2)
{
return arg1 + arg2;
}

template<typename TAcc, typename FP>
ALPAKA_FN_ACC auto almost_equal(TAcc const& acc, FP x, FP y, int ulp)
-> std::enable_if_t<!std::numeric_limits<FP>::is_integer, bool>
{
return alpaka::math::abs(acc, x - y) <= std::numeric_limits<FP>::epsilon()
* alpaka::math::abs(acc, x + y) * static_cast<FP>(ulp)
|| alpaka::math::abs(acc, x - y) < std::numeric_limits<FP>::min();
}

template<typename TAcc, typename FP>
ALPAKA_FN_ACC bool almost_equal(TAcc const& acc, alpaka::Complex<FP> x, alpaka::Complex<FP> y, int ulp)
{
return almost_equal(acc, x.real(), y.real(), ulp) && almost_equal(acc, x.imag(), y.imag(), ulp);
}

} 
} 
} 
} 
