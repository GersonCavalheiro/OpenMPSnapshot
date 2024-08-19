
#pragma once

#include <string>
#include <vector>

#include "containers/container_expression/expressions/expression.h"

namespace Kratos {



template <class TDataType>
class KRATOS_API(KRATOS_CORE) LiteralExpression : public Expression {
public:

LiteralExpression(
const TDataType& Value,
const IndexType NumberOfEntities);


static Expression::Pointer Create(
const TDataType& Value,
const IndexType NumberOfEntities);

double Evaluate(
const IndexType EntityIndex,
const IndexType EntityDataBeginIndex,
const IndexType ComponentIndex) const override;

const std::vector<IndexType> GetItemShape() const override;

std::string Info() const override;

private:

const TDataType mValue;

std::vector<IndexType> mShape;

};

} 