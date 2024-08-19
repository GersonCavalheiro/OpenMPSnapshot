
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "containers/container_expression/expressions/expression.h"

namespace Kratos {


namespace BinaryOperations
{
struct Addition       { static inline constexpr double Evaluate(const double V1, const double V2) { return V1 + V2; } };
struct Substraction   { static inline constexpr double Evaluate(const double V1, const double V2) { return V1 - V2; } };
struct Multiplication { static inline constexpr double Evaluate(const double V1, const double V2) { return V1 * V2; } };
struct Division       { static inline constexpr double Evaluate(const double V1, const double V2) { return V1 / V2; } };
struct Power          { static inline           double Evaluate(const double V1, const double V2) { return std::pow(V1, V2); } };
}


template <class TOperationType>
class KRATOS_API(KRATOS_CORE) BinaryExpression : public Expression {
public:

using IndexType = std::size_t;


BinaryExpression(
Expression::Pointer pLeft,
Expression::Pointer pRight);


static Expression::Pointer Create(
Expression::Pointer pLeft,
Expression::Pointer pRight);

double Evaluate(
const IndexType EntityIndex,
const IndexType EntityDataBeginIndex,
const IndexType ComponentIndex) const override;

const std::vector<IndexType> GetItemShape() const override;

std::string Info() const override;

protected:

const Expression::Pointer mpLeft;

const Expression::Pointer mpRight;

};

} 