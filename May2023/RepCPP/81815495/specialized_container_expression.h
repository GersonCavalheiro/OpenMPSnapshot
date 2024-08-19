
#pragma once

#include <string>

#include "includes/define.h"
#include "includes/model_part.h"
#include "containers/container_expression/container_expression.h"

namespace Kratos {



template <class TContainerType, class TContainerDataIO, class TMeshType = MeshType::Local>
class SpecializedContainerExpression : public ContainerExpression<TContainerType, TMeshType> {
public:

using BaseType = ContainerExpression<TContainerType, TMeshType>;

using IndexType = std::size_t;

KRATOS_CLASS_POINTER_DEFINITION(SpecializedContainerExpression);


SpecializedContainerExpression(ModelPart& rModelPart)
: BaseType(rModelPart)
{
}

SpecializedContainerExpression(const BaseType& rOther)
: BaseType(rOther)
{
}

SpecializedContainerExpression& operator=(const SpecializedContainerExpression& rOther);



SpecializedContainerExpression::Pointer Clone() const;

using BaseType::Read;

using BaseType::MoveFrom;


template <class TDataType>
void Read(
const Variable<TDataType>& rVariable);

using BaseType::Evaluate;


template <class TDataType>
void Evaluate(
const Variable<TDataType>& rVariable);


template <class TDataType>
void SetData(
const TDataType& rValue);


template <class TDataType>
void SetZero(
const Variable<TDataType>& rVariable);


SpecializedContainerExpression operator+(const SpecializedContainerExpression& rOther) const;

SpecializedContainerExpression& operator+=(const SpecializedContainerExpression& rOther);

SpecializedContainerExpression operator+(const double Value) const;

SpecializedContainerExpression& operator+=(const double Value);

SpecializedContainerExpression operator-(const SpecializedContainerExpression& rOther) const;

SpecializedContainerExpression& operator-=(const SpecializedContainerExpression& rOther);

SpecializedContainerExpression operator-(const double Value) const;

SpecializedContainerExpression& operator-=(const double Value);

SpecializedContainerExpression operator*(const SpecializedContainerExpression& rOther) const;

SpecializedContainerExpression& operator*=(const SpecializedContainerExpression& rOther);

SpecializedContainerExpression operator*(const double Value) const;

SpecializedContainerExpression& operator*=(const double Value);

SpecializedContainerExpression operator/(const SpecializedContainerExpression& rOther) const;

SpecializedContainerExpression& operator/=(const SpecializedContainerExpression& rOther);

SpecializedContainerExpression operator/(const double Value) const;

SpecializedContainerExpression& operator/=(const double Value);

SpecializedContainerExpression Pow(const SpecializedContainerExpression& rOther) const;

SpecializedContainerExpression Pow(const double Value) const;


std::string Info() const override;

};

template <class TContainerType, class TContainerDataIO, class TMeshType>
inline std::ostream& operator<<(
std::ostream& rOStream,
const SpecializedContainerExpression<TContainerType, TContainerDataIO, TMeshType>& rThis)
{
return rOStream << rThis.Info();
}

} 

#include "specialized_container_expression_impl.h"
