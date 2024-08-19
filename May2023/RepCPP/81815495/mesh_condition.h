
#pragma once



#include "includes/condition.h"

namespace Kratos
{

class MeshCondition
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

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( MeshCondition);


public:



MeshCondition(IndexType NewId = 0);


MeshCondition(
IndexType NewId,
const NodesArrayType& rThisNodes
);


MeshCondition(
IndexType NewId,
GeometryType::Pointer pGeometry
);


MeshCondition(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties
);

MeshCondition(MeshCondition const& rOther);

~MeshCondition() override;


MeshCondition& operator=(MeshCondition const& rOther);



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


void AddExplicitContribution(
const VectorType& rRHSVector,
const Variable<VectorType>& rRHSVariable,
const Variable<double >& rDestinationVariable,
const ProcessInfo& rCurrentProcessInfo
) override;


void AddExplicitContribution(
const VectorType& rRHS,
const Variable<VectorType>& rRHSVariable,
const Variable<array_1d<double,3> >& rDestinationVariable,
const ProcessInfo& rCurrentProcessInfo
) override;


void AddExplicitContribution(
const MatrixType& rLHSMatrix,
const Variable<MatrixType>& rLHSVariable,
const Variable<Matrix>& rDestinationVariable,
const ProcessInfo& rCurrentProcessInfo
) override;



const Parameters GetSpecifications() const override;

std::string Info() const override
{
std::stringstream buffer;
buffer << "Geometrical Condition #" << Id();
return buffer.str();
}


void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "Geometrical Condition #" << Id();
}

void PrintData(std::ostream& rOStream) const override
{
pGetGeometry()->PrintData(rOStream);
}


private:

friend class Serializer;

void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;


}; 


} 