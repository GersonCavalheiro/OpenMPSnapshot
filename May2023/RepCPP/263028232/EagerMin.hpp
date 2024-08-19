#pragma once

#include "../../lazy/Expr.hpp"
#include "../ExecutionPolicy.hpp"
#include "../serial/SerialMin.hpp"
#include "../parallel/ParallelMin.hpp"

namespace tml
{
namespace eager
{
template<typename Scalar, typename Backend = details::SEQ>
TML_INLINE Scalar min(const tml::matrix<Scalar>& matrix, Backend = tml::execution::seq)
{
Scalar result;
details::backend::min_backend<Scalar, Backend>::min(matrix, result);
return result;
}

template<typename Scalar, typename T, typename Backend = details::SEQ>
TML_INLINE Scalar min(const expr_op<Scalar, T>& expr, Backend = tml::execution::seq)
{
const tml::matrix<Scalar> matrix = expr;
Scalar result;
details::backend::min_backend<Scalar, Backend>::min(matrix, result);
return result;
}

template<typename Scalar, typename Backend = details::SEQ>
TML_INLINE tml::matrix<Scalar> min_rows(const tml::matrix<Scalar>& matrix, Backend = tml::execution::seq)
{
tml::matrix<Scalar> result(shape{ 1, matrix.rows() });
details::backend::min_backend<Scalar, Backend>::rows(matrix, result);
return result;
}

template<typename Scalar, typename T, typename Backend = details::SEQ>
TML_INLINE tml::matrix<Scalar> min_rows(const expr_op<Scalar, T>& expr, Backend = tml::execution::seq)
{
tml::matrix<Scalar> result(shape{ 1, expr.shape.rows });
const tml::matrix<Scalar> matrix = expr;
details::backend::min_backend<Scalar, Backend>::rows(matrix, result);
return result;
}

template<typename Scalar, typename Backend = details::SEQ>
TML_INLINE tml::matrix<Scalar> min_columns(const tml::matrix<Scalar>& matrix, Backend = tml::execution::seq)
{
tml::matrix<Scalar> result(shape{ 1, matrix.columns() });
details::backend::min_backend<Scalar, Backend>::columns(matrix, result);
return result;
}

template<typename Scalar, typename T, typename Backend = details::SEQ>
TML_INLINE tml::matrix<Scalar> min_columns(const expr_op<Scalar, T>& expr, Backend = tml::execution::seq)
{
tml::matrix<Scalar> result(shape{ 1, expr.shape.columns });
const tml::matrix<Scalar> matrix = expr;
details::backend::min_backend<Scalar, Backend>::columns(matrix, result);
return result;
}
}
}