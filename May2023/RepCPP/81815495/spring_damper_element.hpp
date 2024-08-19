
#pragma once



#include "includes/element.h"

namespace Kratos
{

template<std::size_t TDim>
class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) SpringDamperElement
: public Element
{
public:

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( SpringDamperElement);

public:


SpringDamperElement(IndexType NewId, GeometryType::Pointer pGeometry);

SpringDamperElement(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties);

SpringDamperElement(SpringDamperElement const& rOther);

~SpringDamperElement() override;


SpringDamperElement& operator=(SpringDamperElement const& rOther);



Element::Pointer Create(IndexType NewId, NodesArrayType const& ThisNodes, PropertiesType::Pointer pProperties) const override;


Element::Pointer Create(
IndexType NewId,
GeometryType::Pointer pGeom,
PropertiesType::Pointer pProperties
) const override;


Element::Pointer Clone(IndexType NewId, NodesArrayType const& ThisNodes) const override;

/
void GetDofList(DofsVectorType& rElementalDofList, const ProcessInfo& rCurrentProcessInfo) const override;


void EquationIdVector(EquationIdVectorType& rResult, const ProcessInfo& rCurrentProcessInfo) const override;


void GetValuesVector(Vector& rValues, int Step = 0) const override;


void GetFirstDerivativesVector(Vector& rValues, int Step = 0) const override;


void GetSecondDerivativesVector(Vector& rValues, int Step = 0) const override;


/

void CalculateLocalSystem(MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;




void CalculateRightHandSide( VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;



void CalculateLeftHandSide( MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override;


void CalculateMassMatrix(MatrixType& rMassMatrix,
const ProcessInfo& rCurrentProcessInfo) override;


void CalculateDampingMatrix(MatrixType& rDampingMatrix,
const ProcessInfo& rCurrentProcessInfo) override;


int Check(const ProcessInfo& rCurrentProcessInfo) const override;


protected:

static constexpr unsigned int msNumNodes = 2;
static constexpr unsigned int msLocalSize = (TDim == 2) ? 3 : (TDim == 3) ? 6 : 0;
static constexpr unsigned int msElementSize = msLocalSize * msNumNodes;


SpringDamperElement() : Element()
{
}


private:


void ConstCalculateDampingMatrix(MatrixType& rDampingMatrix) const;

void ConstCalculateLocalSystem(MatrixType& rLeftHandSideMatrix, VectorType& rRightHandSideVector) const;

void ConstCalculateLeftHandSide(MatrixType& rLeftHandSideMatrix) const;

void ConstCalculateRightHandSide(VectorType& rRightHandSideVector) const;


friend class Serializer;


void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;



}; 


} 
