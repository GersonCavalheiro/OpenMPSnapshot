
#pragma once

#include <vector>
#include <type_traits>

#include "containers/container_expression/expressions/expression.h"
#include "containers/container_expression/expressions/literal/literal_flat_expression.h"

namespace Kratos {



template<class TDataType>
class KRATOS_API(KRATOS_CORE) VariableExpressionDataIO
{
public:

using Pointer = std::shared_ptr<VariableExpressionDataIO<TDataType>>;

using IndexType = std::size_t;

using RawType = std::conditional_t<std::is_same_v<TDataType, int>, int, double>;

using RawLiteralFlatExpression = LiteralFlatExpression<RawType>;


VariableExpressionDataIO(const TDataType& SampleValue);

VariableExpressionDataIO(const std::vector<IndexType>& rShape);


static Pointer Create(const TDataType& SampleValue);

static Pointer Create(const std::vector<IndexType>& rShape);

void Assign(
TDataType& rOutput,
const Expression& rExpression,
const IndexType EntityIndex) const;

void Read(
RawLiteralFlatExpression& rExpression,
const IndexType EntityIndex,
const TDataType& Value) const;

const std::vector<IndexType> GetItemShape() const { return mShape; }

private:

std::vector<IndexType> mShape;

};

} 