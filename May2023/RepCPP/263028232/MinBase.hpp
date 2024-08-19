#pragma once

#include <numeric>
#include <limits>
#include "Base.hpp"

namespace tml
{
namespace eager
{
namespace details
{
namespace backend
{
template<typename Scalar, typename Backend>
struct min_backend
{
TML_STRONG_INLINE void min(const tml::matrix<Scalar>& matrix, Scalar& result)
{
TML_UNSUPPORTED_BACKEND(Backend);
}

TML_STRONG_INLINE void rows(const tml::matrix<Scalar>& matrix, tml::matrix<Scalar>& result)
{
TML_UNSUPPORTED_BACKEND(Backend);
}

TML_STRONG_INLINE void columns(const tml::matrix<Scalar>& matrix, tml::matrix<Scalar>& result)
{
TML_UNSUPPORTED_BACKEND(Backend);
}
};
}
}
}
}
