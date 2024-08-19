
#pragma once



#include "includes/element.h"

namespace Kratos
{

class MeshElement
: public Element
{
public:


typedef Element BaseType;

typedef BaseType::IndexType IndexType;

typedef BaseType::SizeType SizeType;

typedef BaseType::NodeType NodeType;

typedef BaseType::PropertiesType PropertiesType;

typedef BaseType::GeometryType GeometryType;

typedef BaseType::NodesArrayType NodesArrayType;

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( MeshElement);


public:



MeshElement(IndexType NewId = 0);


MeshElement(
IndexType NewId,
const NodesArrayType& rThisNodes
);


MeshElement(
IndexType NewId,
GeometryType::Pointer pGeometry
);


MeshElement(
IndexType NewId,
GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties
);

MeshElement(MeshElement const& rOther);

~MeshElement() override;


MeshElement& operator=(MeshElement const& rOther);



Element::Pointer Create(
IndexType NewId,
NodesArrayType const& ThisNodes,
PropertiesType::Pointer pProperties
) const override;


Element::Pointer Create(
IndexType NewId,
GeometryType::Pointer pGeom,
PropertiesType::Pointer pProperties
) const override;


Element::Pointer Clone (
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


void CalculateOnIntegrationPoints(
const Variable<bool>& rVariable,
std::vector<bool>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<int>& rVariable,
std::vector<int>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<double>& rVariable,
std::vector<double>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<array_1d<double, 3 > >& rVariable,
std::vector< array_1d<double, 3 > >& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<array_1d<double, 6 > >& rVariable,
std::vector< array_1d<double, 6 > >& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<Vector >& rVariable,
std::vector< Vector >& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<Matrix >& rVariable,
std::vector< Matrix >& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;


void CalculateOnIntegrationPoints(
const Variable<ConstitutiveLaw::Pointer>& rVariable,
std::vector<ConstitutiveLaw::Pointer>& rOutput,
const ProcessInfo& rCurrentProcessInfo
) override;



const Parameters GetSpecifications() const override;

std::string Info() const override
{
std::stringstream buffer;
buffer << "Geometrical Element #" << Id();
return buffer.str();
}


void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "Geometrical  Element #" << Id();
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
