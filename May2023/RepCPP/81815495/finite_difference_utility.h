
#pragma once



#include "includes/define.h"
#include "includes/element.h"
#include "includes/condition.h"
#include "structural_mechanics_application_variables.h"

namespace Kratos
{




class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) FiniteDifferenceUtility
{
public:

typedef Variable<double> array_1d_component_type;
typedef std::size_t IndexType;
typedef std::size_t SizeType;

static void CalculateRightHandSideDerivative(Element& rElement,
const Vector& rRHS,
const Variable<double>& rDesignVariable,
const double& rPertubationSize,
Matrix& rOutput,
const ProcessInfo& rCurrentProcessInfo);

template <typename TElementType>
static void CalculateRightHandSideDerivative(TElementType& rElement,
const Vector& rRHS,
const array_1d_component_type& rDesignVariable,
Node& rNode,
const double& rPertubationSize,
Vector& rOutput,
const ProcessInfo& rCurrentProcessInfo)
{
KRATOS_TRY;

if( rDesignVariable == SHAPE_SENSITIVITY_X || rDesignVariable == SHAPE_SENSITIVITY_Y || rDesignVariable == SHAPE_SENSITIVITY_Z )
{
const IndexType coord_dir =
FiniteDifferenceUtility::GetCoordinateDirection(rDesignVariable);

Vector RHS_perturbed;

if (rOutput.size() != rRHS.size())
rOutput.resize(rRHS.size(), false);

rNode.GetInitialPosition()[coord_dir] += rPertubationSize;
rNode.Coordinates()[coord_dir] += rPertubationSize;

rElement.CalculateRightHandSide(RHS_perturbed, rCurrentProcessInfo);

noalias(rOutput) = (RHS_perturbed - rRHS) / rPertubationSize;

rNode.GetInitialPosition()[coord_dir] -= rPertubationSize;
rNode.Coordinates()[coord_dir] -= rPertubationSize;
}
else
{
KRATOS_WARNING("FiniteDifferenceUtility") << "Unsupported nodal design variable: " << rDesignVariable << std::endl;
if ( (rOutput.size() != 0) )
rOutput.resize(0,false);
}

KRATOS_CATCH("");
}

static void CalculateLeftHandSideDerivative(Element& rElement,
const Matrix& rLHS,
const array_1d_component_type& rDesignVariable,
Node& rNode,
const double& rPertubationSize,
Matrix& rOutput,
const ProcessInfo& rCurrentProcessInfo);

static void CalculateMassMatrixDerivative(Element& rElement,
const Matrix& rMassMatrix,
const array_1d_component_type& rDesignVariable,
Node& rNode,
const double& rPertubationSize,
Matrix& rOutput,
const ProcessInfo& rCurrentProcessInfo);

private:

static std::size_t GetCoordinateDirection(const array_1d_component_type& rDesignVariable);

}; 



}  


