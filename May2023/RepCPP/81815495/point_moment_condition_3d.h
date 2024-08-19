
#pragma once



#include "custom_conditions/base_load_condition.h"

namespace Kratos
{






class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION)  PointMomentCondition3D
: public BaseLoadCondition
{
public:


KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( PointMomentCondition3D );


PointMomentCondition3D(
IndexType NewId,
GeometryType::Pointer pGeometry
);

PointMomentCondition3D(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties
);

~PointMomentCondition3D() override;





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


void EquationIdVector(
EquationIdVectorType& rResult,
const ProcessInfo& rCurrentProcessInfo
) const override;


void GetDofList(
DofsVectorType& ElementalDofList,
const ProcessInfo& rCurrentProcessInfo
) const override;


void GetValuesVector(
Vector& rValues,
int Step = 0
) const override;


void GetFirstDerivativesVector(
Vector& rValues,
int Step = 0
) const override;


void GetSecondDerivativesVector(
Vector& rValues,
int Step = 0
) const override;



int Check( const ProcessInfo& rCurrentProcessInfo ) const override;







std::string Info() const override
{
std::stringstream buffer;
buffer << "PointMomentCondition3D #" << Id();
return buffer.str();
}


void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "PointMomentCondition3D #" << Id();
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


virtual double GetPointMomentIntegrationWeight() const;




PointMomentCondition3D() {};

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
