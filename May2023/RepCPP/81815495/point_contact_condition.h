
#pragma once



#include "custom_conditions/base_load_condition.h"

namespace Kratos
{








class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION)  PointContactCondition
: public BaseLoadCondition
{
public:

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( PointContactCondition );


PointContactCondition(
IndexType NewId,
GeometryType::Pointer pGeometry
);

PointContactCondition(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties
);

~PointContactCondition() override;





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
buffer << "PointContactCondition #" << Id();
return buffer.str();
}


void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "PointContactCondition #" << Id();
}

void PrintData(std::ostream& rOStream) const override
{
pGetGeometry()->PrintData(rOStream);
}




protected:








void CalculateAll(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo,
const bool CalculateStiffnessMatrixFlag,
const bool CalculateResidualVectorFlag
) override;


virtual double GetPointLoadIntegrationWeight() const;






PointContactCondition() {};


private:











friend class Serializer;

void save( Serializer& rSerializer ) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, BaseLoadCondition );
}

void load( Serializer& rSerializer ) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, BaseLoadCondition );
}






}; 








}  


