#pragma once

#include "../LazyOPs.hpp"
#include "../Expr.hpp"
#include "../UnaryOP.hpp"
#include "../ScalarExpr.hpp"
#include "../../../matrix/Matrix.hpp"

namespace tml
{
namespace lazy
{
template<typename Scalar>
Scalar cosh(Scalar scalar)
{
return cosh_op<Scalar>::op(scalar);
}

template<typename Scalar>
expr_op<Scalar, unary_op<Scalar, typename tml::matrix<Scalar>::const_iterator, cosh_op<Scalar>>>
cosh(const tml::matrix<Scalar>& matrix)
{
typedef unary_op<Scalar, typename tml::matrix<Scalar>::const_iterator, cosh_op<Scalar>> ExprType;
return expr_op<Scalar, ExprType>(ExprType(matrix.cbegin()), matrix.get_shape());
}

template<typename Scalar, typename T>
expr_op<Scalar, unary_op<Scalar, expr_op<Scalar, T>, cosh_op<Scalar>>>
cosh(const expr_op<Scalar, T>& expr)
{
typedef unary_op<Scalar, expr_op<Scalar, T>, cosh_op<Scalar>> ExprType;
return expr_op<Scalar, ExprType>(ExprType(expr), expr.shape);
}
}
}