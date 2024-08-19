
#pragma once

#include <cmath>
#include <string>
#include <variant>
#include <vector>
#include <optional>

#include "includes/define.h"
#include "includes/model_part.h"
#include "containers/container_expression/expressions/expression.h"

namespace Kratos {


namespace MeshType {
struct Local     {};
struct Ghost     {};
struct Interface {};
} 


template <class TContainerType, class TMeshType = MeshType::Local>
class KRATOS_API(KRATOS_CORE) ContainerExpression {
public:

using IndexType = std::size_t;

KRATOS_CLASS_POINTER_DEFINITION(ContainerExpression);


virtual ~ContainerExpression() = default;



void CopyFrom(const ContainerExpression<TContainerType, TMeshType>& rOther);


void Read(
double const* pBegin,
const int NumberOfEntities,
int const* pShapeBegin,
const int ShapeSize);


void Read(
int const* pBegin,
const int NumberOfEntities,
int const* pShapeBegin,
const int ShapeSize);


void MoveFrom(
double* pBegin,
const int NumberOfEntities,
int const* pShapeBegin,
const int ShapeSize);


void MoveFrom(
int* pBegin,
const int NumberOfEntities,
int const* pShapeBegin,
const int ShapeSize);


void Evaluate(
double* pBegin,
const int NumberOfEntities,
int const* pShapeBegin,
const int ShapeSize) const;


void SetDataToZero();



void SetExpression(Expression::Pointer pExpression);


bool HasExpression() const;


const Expression& GetExpression() const;


const Expression::Pointer pGetExpression() const;


const std::vector<IndexType> GetItemShape() const;


IndexType GetItemComponentCount() const;


ModelPart& GetModelPart();


const ModelPart& GetModelPart() const;


TContainerType& GetContainer();


const TContainerType& GetContainer() const;


virtual std::string Info() const;


std::string PrintData() const;

protected:

ContainerExpression(ModelPart& rModelPart);

ContainerExpression(const ContainerExpression& rOther);


std::optional<Expression::Pointer> mpExpression;

ModelPart* const mpModelPart;

};


template<class TContainerType>
inline std::ostream& operator<<(
std::ostream& rOStream,
const ContainerExpression<TContainerType>& rThis)
{
return rOStream << rThis.Info();
}


} 
