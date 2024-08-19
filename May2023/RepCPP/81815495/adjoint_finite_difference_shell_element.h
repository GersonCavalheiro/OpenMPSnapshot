
#pragma once




#include "adjoint_finite_difference_base_element.h"

namespace Kratos
{







template <typename TPrimalElement>
class AdjointFiniteDifferencingShellElement
: public AdjointFiniteDifferencingBaseElement<TPrimalElement>
{
public:


typedef AdjointFiniteDifferencingBaseElement<TPrimalElement> BaseType;
typedef typename BaseType::SizeType SizeType;
typedef typename BaseType::IndexType IndexType;
typedef typename BaseType::GeometryType GeometryType;
typedef typename BaseType::PropertiesType PropertiesType;
typedef typename BaseType::NodesArrayType NodesArrayType;
typedef typename BaseType::VectorType VectorType;
typedef typename BaseType::MatrixType MatrixType;
typedef typename BaseType::EquationIdVectorType EquationIdVectorType;
typedef typename BaseType::DofsVectorType DofsVectorType;
typedef typename BaseType::DofsArrayType DofsArrayType;
typedef typename BaseType::IntegrationMethod IntegrationMethod;
typedef typename BaseType::GeometryDataType GeometryDataType;

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(AdjointFiniteDifferencingShellElement);




AdjointFiniteDifferencingShellElement(IndexType NewId = 0)
: BaseType(NewId, true)
{
}

AdjointFiniteDifferencingShellElement(IndexType NewId, typename GeometryType::Pointer pGeometry)
: BaseType(NewId, pGeometry, true)
{
}

AdjointFiniteDifferencingShellElement(IndexType NewId,
typename GeometryType::Pointer pGeometry,
typename PropertiesType::Pointer pProperties)
: BaseType(NewId, pGeometry, pProperties, true)
{
}




Element::Pointer Create(IndexType NewId,
NodesArrayType const& ThisNodes,
typename PropertiesType::Pointer pProperties) const override
{
return Kratos::make_intrusive<AdjointFiniteDifferencingShellElement<TPrimalElement>>(
NewId, this->GetGeometry().Create(ThisNodes), pProperties);
}

Element::Pointer Create(IndexType NewId,
typename GeometryType::Pointer pGeometry,
typename PropertiesType::Pointer pProperties) const override
{
return Kratos::make_intrusive<AdjointFiniteDifferencingShellElement<TPrimalElement>>(
NewId, pGeometry, pProperties);
}

int Check(const ProcessInfo& rCurrentProcessInfo) const override;



protected:



private:



void CheckDofs() const;
void CheckProperties(const ProcessInfo& rCurrentProcessInfo) const;
void CheckSpecificProperties() const;

double GetPerturbationSizeModificationFactor(const Variable<array_1d<double,3>>& rDesignVariable) const override;






friend class Serializer;

void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;





};

}
