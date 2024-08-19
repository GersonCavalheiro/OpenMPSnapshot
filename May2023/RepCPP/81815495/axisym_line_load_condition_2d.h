
#pragma once



#include "custom_conditions/line_load_condition.h"

namespace Kratos
{






class AxisymLineLoadCondition2D
: public LineLoadCondition<2>
{
public:

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(AxisymLineLoadCondition2D);


AxisymLineLoadCondition2D(IndexType NewId, GeometryType::Pointer pGeometry);
AxisymLineLoadCondition2D(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties);

~AxisymLineLoadCondition2D() override;



Condition::Pointer Create(
IndexType NewId,
NodesArrayType const& ThisNodes,
PropertiesType::Pointer pProperties
) const override;


Condition::Pointer Create(
IndexType NewId,
GeometryType::Pointer pGeom,
PropertiesType::Pointer pProperties
) const override;


Condition::Pointer Clone (
IndexType NewId,
NodesArrayType const& ThisNodes
) const override;



std::string Info() const override
{
std::stringstream buffer;
buffer << "AxisymLineLoadCondition2D #" << Id();
return buffer.str();
}


void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "AxisymLineLoadCondition2D #" << Id();
}

void PrintData(std::ostream& rOStream) const override
{
pGetGeometry()->PrintData(rOStream);
}


protected:


AxisymLineLoadCondition2D() : LineLoadCondition<2>()
{
}


private:





double GetIntegrationWeight(
const GeometryType::IntegrationPointsArrayType& IntegrationPoints,
const SizeType PointNumber,
const double detJ
) const override;



friend class Serializer;


void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;


}; 


} 
