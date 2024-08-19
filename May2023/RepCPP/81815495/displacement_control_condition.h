
#pragma once



#include "includes/condition.h"

namespace Kratos
{







class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION)  DisplacementControlCondition
: public Condition
{
public:


typedef Condition                BaseType;
typedef BaseType::IndexType      IndexType;
typedef BaseType::SizeType       SizeType;
typedef BaseType::NodeType       NodeType;
typedef BaseType::PropertiesType PropertiesType;
typedef BaseType::GeometryType   GeometryType;
typedef BaseType::NodesArrayType NodesArrayType;

static constexpr double ZeroTolerance = std::numeric_limits<double>::epsilon();

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( DisplacementControlCondition );


DisplacementControlCondition( IndexType NewId = 0 );

DisplacementControlCondition( IndexType NewId, const NodesArrayType& rThisNodes);
DisplacementControlCondition( IndexType NewId, GeometryType::Pointer pGeometry );

DisplacementControlCondition( IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties );

DisplacementControlCondition(DisplacementControlCondition const& rOther);

~DisplacementControlCondition() override;


DisplacementControlCondition& operator=(DisplacementControlCondition const& rOther);



Condition::Pointer Create(
IndexType NewId,
NodesArrayType const& rThisNodes,
PropertiesType::Pointer pProperties
) const override;


Condition::Pointer Create(
IndexType NewId,
GeometryType::Pointer pGeom,
PropertiesType::Pointer pProperties
) const override;


Condition::Pointer Clone (
IndexType NewId,
NodesArrayType const& rThisNodes
) const override;


void EquationIdVector(
EquationIdVectorType& rResult,
const ProcessInfo& rCurrentProcessInfo
) const override;


void GetDofList(
DofsVectorType& rElementalDofList,
const ProcessInfo& rCurrentProcessInfo
) const override;


void GetValuesVector(
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


int Check( const ProcessInfo& rCurrentProcessInfo ) const override;


unsigned int GetBlockSize() const
{
return 2;
}




const Parameters GetSpecifications() const override;

std::string Info() const override
{
std::stringstream buffer;
buffer << "Displacement Control Condition #" << Id();
return buffer.str();
}


void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "Displacement Control Condition #" << Id();
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




private:




Variable<double>* GetDisplacementInDirection() const;
Variable<double>* GetPointLoadInDirection() const;





friend class Serializer;
void save( Serializer& rSerializer ) const override;
void load( Serializer& rSerializer ) override;

}; 




} 
