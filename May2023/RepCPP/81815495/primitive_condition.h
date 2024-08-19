
#pragma once





#include "wave_condition.h"

namespace Kratos
{





template<std::size_t TNumNodes>
class PrimitiveCondition : public WaveCondition<TNumNodes>
{
public:

typedef std::size_t IndexType;

typedef Node NodeType;

typedef Geometry<NodeType> GeometryType;

typedef WaveCondition<TNumNodes> BaseType;

typedef typename BaseType::NodesArrayType NodesArrayType;

typedef typename BaseType::PropertiesType PropertiesType;

typedef typename BaseType::EquationIdVectorType EquationIdVectorType;

typedef typename BaseType::DofsVectorType DofsVectorType;

typedef typename BaseType::ConditionData ConditionData;

typedef typename BaseType::LocalVectorType LocalVectorType;

typedef typename BaseType::LocalMatrixType LocalMatrixType;


KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(PrimitiveCondition);



PrimitiveCondition() : BaseType(){}


PrimitiveCondition(IndexType NewId, const NodesArrayType& ThisNodes) : BaseType(NewId, ThisNodes){}


PrimitiveCondition(IndexType NewId, GeometryType::Pointer pGeometry) : BaseType(NewId, pGeometry){}


PrimitiveCondition(IndexType NewId, GeometryType::Pointer pGeometry, typename PropertiesType::Pointer pProperties) : BaseType(NewId, pGeometry, pProperties){}


~ PrimitiveCondition() override {};



Condition::Pointer Create(IndexType NewId, NodesArrayType const& ThisNodes, typename PropertiesType::Pointer pProperties) const override
{
return Kratos::make_intrusive<PrimitiveCondition<TNumNodes>>(NewId, this->GetGeometry().Create(ThisNodes), pProperties);
}


Condition::Pointer Create(IndexType NewId, GeometryType::Pointer pGeom, typename PropertiesType::Pointer pProperties) const override
{
return Kratos::make_intrusive<PrimitiveCondition<TNumNodes>>(NewId, pGeom, pProperties);
}


Condition::Pointer Clone(IndexType NewId, NodesArrayType const& ThisNodes) const override
{
Condition::Pointer p_new_elem = Create(NewId, this->GetGeometry().Create(ThisNodes), this->pGetProperties());
p_new_elem->SetData(this->GetData());
p_new_elem->Set(Flags(*this));
return p_new_elem;
}


std::string Info() const override
{
return "PrimitiveCondition";
}




protected:

static constexpr IndexType mLocalSize = BaseType::mLocalSize;


void CalculateGaussPointData(
ConditionData& rData,
const IndexType PointIndex,
const array_1d<double,TNumNodes>& rN) override;


private:





friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, Condition);
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, Condition);
}




}; 







}  
