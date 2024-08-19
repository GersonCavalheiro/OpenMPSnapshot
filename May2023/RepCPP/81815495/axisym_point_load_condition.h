
#pragma once



#include "custom_conditions/point_load_condition.h"

namespace Kratos
{






class AxisymPointLoadCondition
: public PointLoadCondition
{
public:

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(AxisymPointLoadCondition);


AxisymPointLoadCondition(IndexType NewId, GeometryType::Pointer pGeometry);
AxisymPointLoadCondition(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties);

~AxisymPointLoadCondition() override;



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
buffer << "AxisymPointLoadCondition #" << Id();
return buffer.str();
}


void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "AxisymPointLoadCondition #" << Id();
}

void PrintData(std::ostream& rOStream) const override
{
pGetGeometry()->PrintData(rOStream);
}


protected:


AxisymPointLoadCondition() : PointLoadCondition()
{
}


private:





double GetPointLoadIntegrationWeight() const override;



friend class Serializer;


void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;


}; 


} 
