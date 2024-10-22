#pragma once

#include "../base/ArgminBase.hpp"

namespace tml
{
namespace eager
{
namespace details
{
namespace backend
{
template<typename Scalar>
struct argmin_backend<Scalar, SEQ>
{
TML_STRONG_INLINE void argmin(const tml::matrix<Scalar>& matrix, Scalar& result)
{
TML_LOG_BACKEND("seq");
result = static_cast<Scalar>(std::min_element(matrix.cbegin(), matrix.cend()) - matrix.cbegin());
}

TML_STRONG_INLINE void rows(const tml::matrix<Scalar>& matrix, tml::matrix<Scalar>& result)
{
TML_LOG_BACKEND("seq");
size_t rows = matrix.rows(), cols = matrix.columns();
for (size_t i = 0; i < rows; ++i)
result[i] = static_cast<Scalar>(std::min_element(matrix.cbegin() + i*cols, matrix.cbegin() + (i + 1)*cols) - (matrix.cbegin() + i*cols));
}

TML_STRONG_INLINE void columns(const tml::matrix<Scalar>& matrix, tml::matrix<Scalar>& result)
{
TML_LOG_BACKEND("seq");
size_t rows = matrix.rows(), cols = matrix.columns();
for (size_t j = 0; j < cols; ++j)
{
Scalar colArgmin = static_cast<Scalar>(0);
for (size_t i = 0; i < rows; ++i)
if (matrix[j + i*cols] < matrix[j + colArgmin*cols])
colArgmin = static_cast<Scalar>(i);
result[j] = colArgmin;
}
}
};
}
}
}
}
