
#pragma once





#include "includes/condition.h"
#include "@{APP_NAME_LOW}_application_variables.h"


namespace Kratos
{






@{KRATOS_CLASS_TEMPLATE}
class @{KRATOS_NAME_CAMEL} @{KRATOS_CLASS_BASE_HEADER}
{
public:

@{KRATOS_CLASS_LOCAL_FLAGS}


typedef @{KRATOS_CLASS_BASE} BaseType;

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(@{KRATOS_NAME_CAMEL});



@{KRATOS_NAME_CAMEL}(IndexType NewId = 0);


@{KRATOS_NAME_CAMEL}(IndexType NewId, const NodesArrayType& ThisNodes);


@{KRATOS_NAME_CAMEL}(IndexType NewId, GeometryType::Pointer pGeometry);


@{KRATOS_NAME_CAMEL}(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties);


@{KRATOS_NAME_CAMEL}(@{KRATOS_NAME_CAMEL} const& rOther);


~@{KRATOS_NAME_CAMEL}() override;


@{KRATOS_NAME_CAMEL} & operator=(@{KRATOS_NAME_CAMEL} const& rOther);





Condition::Pointer Create(IndexType NewId, NodesArrayType const& ThisNodes, PropertiesType::Pointer pProperties) const override;


Condition::Pointer Create(IndexType NewId, GeometryType::Pointer pGeom, PropertiesType::Pointer pProperties) const override;


Condition::Pointer Clone(IndexType NewId, NodesArrayType const& ThisNodes) const override;


void EquationIdVector(EquationIdVectorType& rResult, const ProcessInfo& CurrentProcessInfo) const override;


void GetDofList(DofsVectorType& rConditionDofList, const ProcessInfo& CurrentProcessInfo) const override;




void CalculateLocalSystem(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;


void CalculateLeftHandSide(MatrixType& rLeftHandSideMatrix, const ProcessInfo& rCurrentProcessInfo) override;


void CalculateRightHandSide(VectorType& rRightHandSideVector, const ProcessInfo& rCurrentProcessInfo) override;


void CalculateFirstDerivativesContributions(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;


void CalculateFirstDerivativesLHS(MatrixType& rLeftHandSideMatrix, const ProcessInfo& rCurrentProcessInfo) override;


void CalculateFirstDerivativesRHS(VectorType& rRightHandSideVector, const ProcessInfo& rCurrentProcessInfo) override;





void CalculateSecondDerivativesContributions(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;


void CalculateSecondDerivativesLHS(
MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override;


void CalculateSecondDerivativesRHS(
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;


void CalculateMassMatrix(MatrixType& rMassMatrix, const ProcessInfo& rCurrentProcessInfo) override;


void CalculateDampingMatrix(MatrixType& rDampingMatrix, const ProcessInfo& rCurrentProcessInfo) override;


int Check(const ProcessInfo& rCurrentProcessInfo) const override;






std::string Info() const override;

void PrintInfo(std::ostream& rOStream) const override;

void PrintData(std::ostream& rOStream) const override;



protected:









private:


@{KRATOS_STATIC_MEMBERS_LIST}


@{KRATOS_MEMBERS_LIST}




friend class Serializer;

void save(Serializer& rSerializer) const override;
void load(Serializer& rSerializer) override;





}; 





} 
