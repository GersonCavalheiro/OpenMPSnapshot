
#pragma once

#include <string>
#include <vector>
#include <variant>

#include "includes/define.h"
#include "includes/model_part.h"
#include "containers/container_expression/specialized_container_expression.h"
#include "containers/container_expression/container_data_io.h"

#include "container_properties_data_io.h"

namespace Kratos {


class KRATOS_API(OPTIMIZATION_APPLICATION) CollectiveExpressions {
public:

using IndexType = std::size_t;

using HistoricalExpressionPointer =  SpecializedContainerExpression<ModelPart::NodesContainerType, ContainerDataIO<ContainerDataIOTags::Historical>>::Pointer;

using NodalExpressionPointer =  SpecializedContainerExpression<ModelPart::NodesContainerType, ContainerDataIO<ContainerDataIOTags::NonHistorical>>::Pointer;

using ConditionExpressionPointer =  SpecializedContainerExpression<ModelPart::ConditionsContainerType, ContainerDataIO<ContainerDataIOTags::NonHistorical>>::Pointer;

using ElementExpressionPointer =  SpecializedContainerExpression<ModelPart::ElementsContainerType, ContainerDataIO<ContainerDataIOTags::NonHistorical>>::Pointer;

using ConditionPropertiesExpressionPointer =  SpecializedContainerExpression<ModelPart::ConditionsContainerType, ContainerDataIO<ContainerDataIOTags::Properties>>::Pointer;

using ElementPropertiesExpressionPointer =  SpecializedContainerExpression<ModelPart::ElementsContainerType, ContainerDataIO<ContainerDataIOTags::Properties>>::Pointer;

using CollectiveExpressionType = std::variant<
HistoricalExpressionPointer,
NodalExpressionPointer,
ConditionExpressionPointer,
ElementExpressionPointer,
ConditionPropertiesExpressionPointer,
ElementPropertiesExpressionPointer>;

using VariableTypes = std::variant<
const Variable<double>*,
const Variable<array_1d<double, 3>>*,
const Variable<array_1d<double, 4>>*,
const Variable<array_1d<double, 6>>*,
const Variable<array_1d<double, 9>>*,
const Variable<Vector>*,
const Variable<Matrix>*>;

KRATOS_CLASS_POINTER_DEFINITION(CollectiveExpressions);


CollectiveExpressions() noexcept = default;



CollectiveExpressions(const std::vector<CollectiveExpressionType>& rExpressionPointersList);

CollectiveExpressions(const CollectiveExpressions& rOther);

~CollectiveExpressions() = default;



CollectiveExpressions Clone() const;


void SetToZero();


void Add(const CollectiveExpressionType& pExpression);


void Add(const CollectiveExpressions& rCollectiveExpression);


void Clear();


void Read(
double const* pBegin,
int const* NumberOfEntities,
int const** pListShapeBegin,
int const* ShapeSizes,
const int NumberOfContainers);


void Read(const VariableTypes& rVariable);


void Read(const std::vector<VariableTypes>& rVariables);


void MoveFrom(
double* pBegin,
int const* NumberOfEntities,
int const** pListShapeBegin,
int const* ShapeSizes,
const int NumberOfContainers);


void Evaluate(
double* pBegin,
const int Size) const;


void Evaluate(const VariableTypes& rVariable);


void Evaluate(const std::vector<VariableTypes>& rVariables);


IndexType GetCollectiveFlattenedDataSize() const;

std::vector<CollectiveExpressionType> GetContainerExpressions();

std::vector<CollectiveExpressionType> GetContainerExpressions() const;

bool IsCompatibleWith(const CollectiveExpressions& rOther) const;


CollectiveExpressions operator+(const CollectiveExpressions& rOther) const;

CollectiveExpressions& operator+=(const CollectiveExpressions& rOther);

CollectiveExpressions operator+(const double Value) const;

CollectiveExpressions& operator+=(const double Value);

CollectiveExpressions operator-(const CollectiveExpressions& rOther) const;

CollectiveExpressions& operator-=(const CollectiveExpressions& rOther);

CollectiveExpressions operator-(const double Value) const;

CollectiveExpressions& operator-=(const double Value);

CollectiveExpressions operator*(const CollectiveExpressions& rOther) const;

CollectiveExpressions& operator*=(const CollectiveExpressions& rOther);

CollectiveExpressions operator*(const double Value) const;

CollectiveExpressions& operator*=(const double Value);

CollectiveExpressions operator/(const CollectiveExpressions& rOther) const;

CollectiveExpressions& operator/=(const CollectiveExpressions& rOther);

CollectiveExpressions operator/(const double Value) const;

CollectiveExpressions& operator/=(const double Value);

CollectiveExpressions Pow(const CollectiveExpressions& rOther) const;

CollectiveExpressions Pow(const double Value) const;


std::string Info() const;

private:

std::vector<CollectiveExpressionType> mExpressionPointersList;

};


inline std::ostream& operator<<(
std::ostream& rOStream,
const CollectiveExpressions& rThis)
{
return rOStream << rThis.Info();
}


} 