
#pragma once



#include "includes/element.h"

namespace Kratos
{

class MassElement : public Element
{
public:


typedef Variable< array_1d< double, 3> > ArrayVariableType;


KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(MassElement);



MassElement(IndexType NewId = 0)
: Element(NewId) {}


MassElement(IndexType NewId, const NodesArrayType& ThisNodes)
: Element(NewId, ThisNodes) {}


MassElement(IndexType NewId, GeometryType::Pointer pGeometry)
: Element(NewId, pGeometry) {}


MassElement(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties)
: Element(NewId, pGeometry, pProperties) {}


MassElement(MassElement const& rOther)
: Element(rOther) {}


~MassElement() override = default;





Element::Pointer Create(IndexType NewId, NodesArrayType const& ThisNodes, PropertiesType::Pointer pProperties) const override;


Element::Pointer Create(IndexType NewId, GeometryType::Pointer pGeom, PropertiesType::Pointer pProperties) const override;


Element::Pointer Clone(IndexType NewId, NodesArrayType const& ThisNodes) const override;


void EquationIdVector(EquationIdVectorType& rResult, const ProcessInfo& CurrentProcessInfo) const override;


void GetDofList(DofsVectorType& rElementalDofList, const ProcessInfo& CurrentProcessInfo) const override;



void GetValuesVector(Vector& values, int Step = 0) const override;


void GetFirstDerivativesVector(Vector& values, int Step = 0) const override;


void GetSecondDerivativesVector(Vector& values, int Step = 0) const override;


void Initialize(const ProcessInfo& rCurrentProcessInfo) override;


void CalculateLocalSystem(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;


void CalculateLeftHandSide(MatrixType& rLeftHandSideMatrix, const ProcessInfo& rCurrentProcessInfo) override;


void CalculateRightHandSide(VectorType& rRightHandSideVector, const ProcessInfo& rCurrentProcessInfo) override;


void CalculateMassMatrix(MatrixType& rMassMatrix, const ProcessInfo& rCurrentProcessInfo) override;


void CalculateDampingMatrix(MatrixType& rDampingMatrix, const ProcessInfo& rCurrentProcessInfo) override;


int Check(const ProcessInfo& rCurrentProcessInfo) const override;


std::string Info() const override;

void PrintInfo(std::ostream& rOStream) const override;

void PrintData(std::ostream& rOStream) const override;


private:


double mMass = 0.0;



void GenericGetValuesVector(Vector& rValues, int Step, const ArrayVariableType& rVariable) const;

double GetElementMass() const;


friend class Serializer;

void save(Serializer& rSerializer) const override;
void load(Serializer& rSerializer) override;


}; 


} 
