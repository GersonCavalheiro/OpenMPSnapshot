
#pragma once



#include "custom_conditions/base_load_condition.h"

namespace Kratos
{







class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION)  SurfaceLoadCondition3D
: public BaseLoadCondition
{
public:


KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( SurfaceLoadCondition3D );


SurfaceLoadCondition3D();

SurfaceLoadCondition3D(
IndexType NewId,
GeometryType::Pointer pGeometry
);

SurfaceLoadCondition3D(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties
);

~SurfaceLoadCondition3D() override;





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


void CalculateOnIntegrationPoints(
const Variable<array_1d<double, 3 > >& rVariable,
std::vector< array_1d<double, 3 > >& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;






std::string Info() const override
{
std::stringstream buffer;
buffer << "Surface load Condition #" << Id();
return buffer.str();
}


void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "Surface load Condition #" << Id();
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


void CalculateAndSubKp(
Matrix& rK,
const array_1d<double, 3>& rTangentXi,
const array_1d<double, 3>& rTangentEta,
const Matrix& rDN_De,
const Vector& rN,
const double Pressure,
const double Weight
) const;


void CalculateAndAddPressureForce(
VectorType& rResidualVector,
const Vector& rN,
const array_1d<double, 3 >& rNormal,
const double Pressure,
const double Weight,
const ProcessInfo& rCurrentProcessInfo
) const;




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
