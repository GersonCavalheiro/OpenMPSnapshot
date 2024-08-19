
#pragma once



#include "includes/condition.h"
#include "structural_mechanics_application_variables.h"

namespace Kratos
{







class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION)  BaseLoadCondition
: public Condition
{
public:


typedef Condition BaseType;

typedef BaseType::IndexType IndexType;

typedef BaseType::SizeType SizeType;

typedef BaseType::NodeType NodeType;

typedef BaseType::PropertiesType PropertiesType;

typedef BaseType::GeometryType GeometryType;

typedef BaseType::NodesArrayType NodesArrayType;

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( BaseLoadCondition );


BaseLoadCondition()
{};

BaseLoadCondition( IndexType NewId, GeometryType::Pointer pGeometry ):Condition(NewId,pGeometry)
{};

BaseLoadCondition( IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties ):Condition(NewId,pGeometry,pProperties)
{};

BaseLoadCondition(BaseLoadCondition const& rOther);

~BaseLoadCondition() override
{};


BaseLoadCondition& operator=(BaseLoadCondition const& rOther);



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


void CalculateLocalSystem(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateRightHandSide(
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateMassMatrix(
MatrixType& rMassMatrix,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateDampingMatrix(
MatrixType& rDampingMatrix,
const ProcessInfo& rCurrentProcessInfo
) override;


void AddExplicitContribution(const VectorType& rRHS,
const Variable<VectorType>& rRHSVariable,
const Variable<array_1d<double,3> >& rDestinationVariable,
const ProcessInfo& rCurrentProcessInfo
) override;


int Check( const ProcessInfo& rCurrentProcessInfo ) const override;


virtual bool HasRotDof() const;


unsigned int GetBlockSize() const
{
unsigned int dim = GetGeometry().WorkingSpaceDimension();
if( HasRotDof() ) { 
if(dim == 2)
return 3;
else if(dim == 3)
return 6;
else
KRATOS_ERROR << "The conditions only works for 2D and 3D elements";
} else {
return dim;
}
}





const Parameters GetSpecifications() const override;

std::string Info() const override
{
std::stringstream buffer;
buffer << "Base load Condition #" << Id();
return buffer.str();
}


void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "Base load Condition #" << Id();
}

void PrintData(std::ostream& rOStream) const override
{
pGetGeometry()->PrintData(rOStream);
}


protected:






virtual void CalculateAll(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo,
const bool CalculateStiffnessMatrixFlag,
const bool CalculateResidualVectorFlag
);


virtual double GetIntegrationWeight(
const GeometryType::IntegrationPointsArrayType& IntegrationPoints,
const SizeType PointNumber,
const double detJ
) const;




private:












friend class Serializer;

void save( Serializer& rSerializer ) const override;

void load( Serializer& rSerializer ) override;

}; 




} 
